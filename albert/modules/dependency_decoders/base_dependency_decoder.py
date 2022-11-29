from __future__ import absolute_import

from typing import List, Optional, Tuple

import numpy
import torch
import torch.nn as nn

import allennlp.nn.util as nn_util
from allennlp.common import Registrable
from allennlp.data import Vocabulary
from allennlp.nn.chu_liu_edmonds import decode_mst


class BaseDependencyDecoder(nn.Module, Registrable):
    def __init__(
        self,
        tag_representation_dim: int,
        vocab: Optional[Vocabulary] = None,
        num_labels: Optional[int] = None,
    ) -> None:
        super(BaseDependencyDecoder, self).__init__()
        self.vocab = vocab
        self._allowed_decoding = set(["greedy", "mst"])
        self._tag_representation_dim = tag_representation_dim
        self._num_labels = (
            num_labels if num_labels else vocab.get_vocab_size("head_tags")
        )

        self.tag_bilinear = torch.nn.modules.Bilinear(
            tag_representation_dim, tag_representation_dim, self._num_labels
        )

    def entropy(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Child class implements this")

    def likelihood(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
        head_indices: torch.LongTensor,
        head_tags: torch.LongTensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Child class implements this")

    def compute_edge_weights_and_labels(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Child class implements this")

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
        raise NotImplementedError("Child class implements this")

    def infer_tree(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
        decoding: str = "greedy",
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        if decoding == "greedy":
            return self.greedy_decoding(
                head_tag_representation, child_tag_representation, arc_scores, mask
            )
        if decoding == "mst":
            return self.mst_decoding(
                head_tag_representation, child_tag_representation, arc_scores, mask
            )
        raise RuntimeError(f"{decoding} not supported now.")

    def greedy_decoding(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        Parameters
        ----------
        head_tag_representation : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : ``torch.Tensor``, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        Returns
        -------
        heads : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : ``torch.Tensor``
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = arc_scores + torch.diag(
            arc_scores.new(mask.size(1)).fill_(-numpy.inf)
        )
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = (1 - mask).to(dtype=torch.bool).unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch, seq_len + 1)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch, seq_len + 1, num_head_tags)
        head_tag_logits = self.get_head_tag_logits(
            head_tag_representation, child_tag_representation, heads
        )
        _, head_tags = head_tag_logits.max(dim=2)
        return heads, head_tags

    def mst_decoding(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        arc_scores: torch.Tensor,
        mask: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        # shape: (batch, seq_len + 1, seq_len + 1), (batch, seq_len + 1, seq_len + 1)
        batch_energy_with_labels = self.compute_edge_weights_and_labels(
            arc_scores=arc_scores,
            head_tag_representation=head_tag_representation,
            child_tag_representation=child_tag_representation,
            mask=mask,
        )
        batch_energy, batch_tag_ids = batch_energy_with_labels.max(1)
        batch_energy = batch_energy.detach().cpu().numpy()
        # shape: (batch,)
        lengths = mask.long().sum(-1).cpu().numpy()

        heads: List[List[int]] = []
        head_tags: List[List[int]] = []
        for scores, length, tag_ids in zip(batch_energy, lengths, batch_tag_ids):
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            # shape: (seq_len + 1, seq_len + 1)
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores, length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs un-batched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return (
            torch.from_numpy(numpy.stack(heads)).long(),
            torch.from_numpy(numpy.stack(head_tags)).long(),
        )

    def get_head_tag_logits(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        head_indices: torch.LongTensor,
    ) -> torch.Tensor:

        batch = head_tag_representation.size(0)
        # shape: (batch,)
        range_vector = nn_util.get_range_vector(
            batch, nn_util.get_device_of(head_tag_representation)
        ).unsqueeze(1)
        # shape: (batch, seq_len + 1, encoding_dim)
        selected_head_tag_representations = head_tag_representation[
            range_vector, head_indices
        ]
        selected_head_tag_representations = (
            selected_head_tag_representations.contiguous()
        )
        # shape: (batch, seq_len + 1, num_head_tags)
        logits = self.tag_bilinear(
            selected_head_tag_representations, child_tag_representation
        )
        return logits
