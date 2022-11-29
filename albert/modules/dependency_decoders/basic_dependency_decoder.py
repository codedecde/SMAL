from __future__ import absolute_import

from typing import Optional

import torch
import torch.nn.functional as F
from overrides import overrides

import albert.nn.util as albert_nn_util
import allennlp.nn.util as nn_util
from albert.modules.dependency_decoders.base_dependency_decoder import (
    BaseDependencyDecoder,
)


@BaseDependencyDecoder.register("basic")
class BasicDependencyDecoder(BaseDependencyDecoder):
    @overrides
    def likelihood(  # pylint: disable=arguments-differ
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
        head_indices: torch.LongTensor,
        head_tags: torch.LongTensor,
        decoding: str = "greedy",
    ) -> torch.Tensor:
        if decoding == "greedy":
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
        if decoding == "mst":
            log_nr = -self.forward(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                arc_scores=arc_scores,
                mask=mask,
                head_indices=head_indices,
                head_tags=head_tags,
                normalize=False,
            )
            # Note that for the basic model, the labels on
            # each arc form a simplex. The log sum exp of
            # the feature labels would be 1. Hence they don't
            # have to be considered while computing the log
            # partition function.
            log_dr = albert_nn_util.non_projective_dependency_partition_function(
                arc_scores, mask
            )
            return log_nr - log_dr
        raise NotImplementedError(f"{decoding} not recognized")

    @overrides
    def entropy(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
        decoding="greedy",
    ) -> torch.Tensor:

        float_mask = mask.float()
        # First compute entropy of labels
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
        # Shape (batch_size, sequence_length + 1, sequence_length + 1, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(
            head_tag_representation, child_tag_representation
        )

        pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=-1)
        label_entropy = (pairwise_head_logits.exp() * pairwise_head_logits).sum(-1)
        label_entropy = (
            label_entropy * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)
        )
        label_entropy = label_entropy[:, 1:, 1:]
        label_entropy = label_entropy.sum(-1).sum(-1)

        if decoding == "greedy":
            # compute the entropy of edges
            # edges are independent of labels
            # first compute entropy of edges
            # shape: (batch, seq_len + 1, seq_len + 1)
            # normalized_arc_scores[b, i, j] = prob(head = j | i) for b^{th} batch

            normalized_arc_scores = (
                nn_util.masked_log_softmax(arc_scores, mask)
                * float_mask.unsqueeze(2)
                * float_mask.unsqueeze(1)
            )
            # shape: (batch, seq_len)
            root_logprobs = normalized_arc_scores[:, 1:, 0]
            root_entropy = (root_logprobs.exp() * root_logprobs).sum(-1)
            # shape: (batch, seq_len, seq_len)
            normalized_arc_scores = normalized_arc_scores[:, 1:, 1:]
            arc_entropy = (
                (normalized_arc_scores.exp() * normalized_arc_scores).sum(-1).sum(-1)
            )
            arc_entropy += root_entropy

        if decoding == "mst":
            raise NotImplementedError("Needs implementation")

        # finally sum
        entropy = -(label_entropy + arc_entropy)
        return entropy

    @overrides
    def forward(  # pylint: disable=arguments-differ
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
        head_tag_logits = self.get_head_tag_logits(
            head_tag_representation=head_tag_representation,
            child_tag_representation=child_tag_representation,
            head_indices=head_indices,
        )
        normalized_head_logits = nn_util.masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * float_mask.unsqueeze(-1)

        batch, seq_len, _ = arc_scores.size()
        normalized_arc_logits = (
            nn_util.masked_log_softmax(arc_scores, mask)
            * float_mask.unsqueeze(2)
            * float_mask.unsqueeze(1)
        )

        batch_indexer = nn_util.get_range_vector(
            batch, device=nn_util.get_device_of(arc_scores)
        ).unsqueeze(1)
        timestep_index = nn_util.get_range_vector(
            seq_len, nn_util.get_device_of(arc_scores)
        )
        child_indices = timestep_index.view(1, seq_len).expand(batch, seq_len).long()
        arc_loss = normalized_arc_logits[batch_indexer, child_indices, head_indices]
        tag_loss = normalized_head_logits[batch_indexer, child_indices, head_tags]

        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]
        if normalize:
            valid_positions = mask.sum() - batch

            arc_nll = -arc_loss.sum() / valid_positions.float()
            tag_nll = -tag_loss.sum() / valid_positions.float()
        else:
            arc_nll = -arc_loss.sum(-1)
            tag_nll = -tag_loss.sum(-1)

        return arc_nll + tag_nll

    @overrides
    def compute_edge_weights_and_labels(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
    ) -> torch.Tensor:
        float_mask = mask.float()
        normalized_arc_scores = (
            nn_util.masked_log_softmax(arc_scores, mask)
            * float_mask.unsqueeze(2)
            * float_mask.unsqueeze(1)
        )
        normalized_arc_scores = normalized_arc_scores.transpose(1, 2)
        # Now construct the labels tag
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
        pairwise_head_logits = self.tag_bilinear(
            head_tag_representation, child_tag_representation
        )

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(
            pairwise_head_logits, dim=3
        ).permute(0, 3, 1, 2)
        batch_energy_with_labels = torch.exp(
            normalized_arc_scores.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return batch_energy_with_labels
