from __future__ import absolute_import

from typing import Optional

import torch
from overrides import overrides

import albert.nn.util as albert_nn_util
import allennlp.nn.util as nn_util
from albert.modules.dependency_decoders.base_dependency_decoder import (
    BaseDependencyDecoder,
)


@BaseDependencyDecoder.register("mst_edges_and_labels")
class MSTEdgesAndLabelsDecoder(BaseDependencyDecoder):
    @overrides
    def likelihood(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
        head_indices: torch.LongTensor,
        head_tags: torch.LongTensor,
    ) -> torch.Tensor:
        neg_log_likelihood = self.forward(
            head_tag_representation=head_tag_representation,
            child_tag_representation=child_tag_representation,
            arc_scores=arc_scores,
            mask=mask,
            head_indices=head_indices,
            head_tags=head_tags,
            normalize=False,
        )
        return -neg_log_likelihood

    def _compute_log_partition(
        self,  # pylint: disable=no-self-use
        batch_energy_with_labels: torch.Tensor,
        mask: torch.LongTensor,
    ) -> torch.Tensor:
        arc_energies = (
            nn_util.logsumexp(batch_energy_with_labels, 1).transpose(1, 2).contiguous()
        )
        log_dr = albert_nn_util.non_projective_dependency_partition_function(
            arc_scores=arc_energies, mask=mask
        )
        return log_dr

    @overrides
    def forward(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
        head_indices: torch.LongTensor,
        head_tags: torch.LongTensor,
        normalize: Optional[bool] = True,
    ) -> torch.Tensor:
        float_mask = mask.float()
        # shape: (batch, seq_len + 1, num_classes)
        head_tag_scores = self.get_head_tag_logits(
            head_tag_representation, child_tag_representation, head_indices
        )
        batch, seq_len, _ = arc_scores.size()
        arc_scores = arc_scores * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
        batch_indexer = nn_util.get_range_vector(
            batch, device=nn_util.get_device_of(arc_scores)
        ).unsqueeze(1)
        timestep_index = nn_util.get_range_vector(
            seq_len, nn_util.get_device_of(arc_scores)
        )
        child_indices = timestep_index.view(1, seq_len).expand(batch, seq_len).long()
        log_nr = (
            arc_scores[batch_indexer, child_indices, head_indices][:, 1:]
            + head_tag_scores[batch_indexer, child_indices, head_tags][:, 1:]
        )
        log_nr = log_nr.sum(-1)

        # now compute the partition function
        batch_energy_with_labels = self.compute_edge_weights_and_labels(
            head_tag_representation, child_tag_representation, arc_scores
        )
        log_dr = self._compute_log_partition(
            batch_energy_with_labels=batch_energy_with_labels, mask=mask
        )
        nll = -(log_nr - log_dr)
        if normalize:
            return nll.mean()
        return nll

    @overrides
    def compute_edge_weights_and_labels(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            float_mask = mask.float()
            arc_scores = arc_scores * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
        arc_scores = arc_scores.transpose(1, 2)
        (
            batch_size,
            sequence_length,
            tag_representation_dim,
        ) = head_tag_representation.size()

        expanded_shape = [
            batch_size,
            sequence_length,
            sequence_length,
            tag_representation_dim,
        ]

        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(
            *expanded_shape
        ).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(
            *expanded_shape
        ).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_tag_scores = self.tag_bilinear(
            head_tag_representation, child_tag_representation
        )

        pairwise_head_tag_scores = pairwise_head_tag_scores.permute(0, 3, 1, 2)
        batch_energy_with_labels = arc_scores.unsqueeze(1) + pairwise_head_tag_scores
        return batch_energy_with_labels
