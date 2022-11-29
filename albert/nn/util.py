from __future__ import absolute_import

from typing import Iterable, List, Optional, Tuple

import torch

import allennlp.nn.util as nn_util


def convert_to_one_hot(vector: torch.LongTensor, num_labels: int) -> torch.LongTensor:
    """Converts a vector into it's one hot format

    ..note ::

        vector (and num_labels = 3)
        [0, 1, 0, 2] -> [[1, 0, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]]

    Parameters
    ----------
    vector : torch.LongTensor
        The vector to convert to one-hot
    num_labels : int
        The maximum number of labels

    Returns
    -------
    torch.LongTensor
        A one-hot vector of shape (batch, num_labels)

    """
    assert (
        vector.ndim == 1
    ), f"Expected tensor of shape: ({vector.size(0)},), got {vector.size()}"
    assert (
        vector.max() < num_labels
    ), f"Can pad from 0 to {num_labels - 1}, found max of {vector.max()} in vector"
    # shape: (batch, num_labels)
    one_hot = torch.zeros(
        (vector.size(0), num_labels), out=torch.empty_like(vector)
    ).long()
    # shape: (batch,)
    range_vector = nn_util.get_range_vector(one_hot.size(0), vector.get_device())
    # shape: (batch, num_labels)
    one_hot[range_vector, vector] = 1
    return one_hot


def select_subset(
    selection_mask: torch.LongTensor, *tensors: torch.Tensor
) -> Iterable[torch.Tensor]:
    """Select a subset of tensors based on a selection mask. This assumes that
    all tensors have the same (batch,) dimension, and that the selection_mask
    indicates which elements to select

    Parameters
    ----------
    selection_mask : torch.Tensor
        The binary selection mask indicating the indices to be selected

    tensors : Iterable[torch.Tensor]
        The tensors to select from

    Returns
    -------
    Iterable[torch.Tensor]
        The index selected tensors
    """
    assert selection_mask.sum() > 0, "Need some indices to select"
    (select_x,) = torch.nonzero(selection_mask, as_tuple=True)
    return (
        torch.index_select(input=tensor, dim=0, index=select_x).contiguous()
        for tensor in tensors
    )


def pad_sequences_to_length(
    sequences: List[List[int]], max_length: int, pad_token: int = 0
) -> List[List[int]]:
    """Given a list of list of int values, pads it to a uniform length, up to max_len

    .. note::
        [
            [1, 2],
            [1, 2, 3]
        ]
        with pad_token = 0 and max_length = 3, gets converted to
        [
            [1, 2, 0],
            [1, 2, 3]
        ]

    Parameters
    ----------
    sequences : List[List[int]]
        The list to pad
    max_length : int
        The length to pad
    pad_token : int, optional (default = 0)
        The token to pad with

    Returns
    -------
    List[List[int]]
        The padded sequences
    """
    new_sequences = []
    for sequence in sequences:
        new_sequence = sequence[:]
        for _ in range(len(sequence), max_length):
            new_sequence.append(pad_token)
        new_sequences.append(new_sequence)
    return new_sequences


def non_projective_dependency_partition_function(
    arc_scores: torch.Tensor,
    mask: torch.LongTensor,
    eps: float = 1e-5,
    single_root_attachment: bool = True,
) -> torch.Tensor:
    """Computes the partition function for a non-projective
    dependency partition structure based on the Matrix-Tree theorem.

    See [1, 2] for details.

    Parameters
    ----------
    arc_scores : torch.Tensor
        The edge scores of shape (batch, seq_len + 1, seq_len + 1)
        arc_scores[i, j, k] = score of the edge k -> j for batch i
    mask : torch.LongTensor
        The mask of shape (batch, seq_len + 1)
        Indicates the presence of the i^{th} word
        (including the root token)
    eps : float, optional (default = 1e-5)
        For numerical stability
    single_root_attachment: bool, Optional (default = ``True'')
        If ``True'' then the set of allowed trees
        have only one root -> Node connection.

    Returns
    -------
    torch.Tensor
        The partition function for each example, of shape (batch,)

    References
    ----------
    .. [1] Terry Koo, Amir Globerson, Xavier Carreras and Michael Collins. "Structured Prediction Models via the Matrix-Tree Theorem."
           Empirical Methods in Natural Language Processing. 2007.
       [2] David A. Smith and Noah A. Smith. "Probabilistic Models of Nonprojective Dependency Trees."
            Empirical Methods in Natural Language Processing. 2007.

    """
    float_mask = mask.float()
    arc_scores = arc_scores * float_mask.unsqueeze(2) * float_mask.unsqueeze(1)

    # shape: (batch, seq_len + 1, seq_len + 1)
    energy_score = arc_scores.exp() + eps
    # shape: (batch, seq_len, seq_len)
    modified_energy_score = energy_score[:, 1:, 1:]
    # shape: (seq_len, seq_len)
    eye = torch.eye(modified_energy_score.size(1), device=energy_score.device)

    laplacian = modified_energy_score
    laplacian = laplacian.masked_fill(eye != 0, 0)

    if single_root_attachment:
        laplacian = -laplacian + torch.diag_embed(
            laplacian.sum(2), offset=0, dim1=-2, dim2=-1
        )
        laplacian[:, :, 0] = energy_score[:, 1:, 0]
    else:
        laplacian = -laplacian + torch.diag_embed(
            laplacian.sum(2) + energy_score[:, 1:, 0], offset=0, dim1=-2, dim2=-1
        )

    partition = laplacian.logdet()
    return partition
