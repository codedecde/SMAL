"""
This is mostly a copy of the Dependency parser from AllenNLP. This is mostly so that we can modify
it for our purposes. It also uses a Bert embedder for as input.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from overrides import overrides
from torch.nn.modules import Dropout

from albert.modules.dependency_decoders.base_dependency_decoder import (
    BaseDependencyDecoder,
)
from allennlp.common.checks import ConfigurationError, check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    Embedding,
    FeedForward,
    InputVariationalDropout,
    Seq2SeqEncoder,
    TextFieldEmbedder,
)
from allennlp.modules.matrix_attention.bilinear_matrix_attention import (
    BilinearMatrixAttention,
)
from allennlp.nn import Activation, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, get_text_field_mask
from allennlp.training.metrics import AttachmentScores

logger = logging.getLogger(__name__)

POS_TO_IGNORE = {"``", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}


@Model.register("custom_dependency_parser")
class CustomBiaffineDependencyParser(Model):
    """
    This dependency parser follows the model of
    ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimal Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    Additionally, for active learning, we want not just the optimal score,
    but also the probability of the best sequence. This variant does just that
    based on
    https://www.aclweb.org/anthology/D07-1015.pdf and https://www.aclweb.org/anthology/D07-1014.pdf

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : ``int``, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : ``int``, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : ``FeedForward``, optional, (default = None).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    pos_tag_embedding : ``Embedding``, optional.
        Used to embed the ``pos_tags`` ``SequenceLabelField`` we get as input to the model.
    use_mst_decoding_for_validation : ``bool``, optional (default = True).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    dropout : ``float``, optional, (default = 0.0)
        The variational dropout applied to the output of the encoder and MLP layers.
    input_dropout : ``float``, optional, (default = 0.0)
        The dropout applied to the embedded text input.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        tag_representation_dim: int,
        arc_representation_dim: int,
        decoder: BaseDependencyDecoder,
        tag_feedforward: FeedForward = None,
        arc_feedforward: FeedForward = None,
        pos_embed_dim: int = None,
        use_mst_decoding_for_validation: bool = True,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        encoder_dim = encoder.get_output_dim()

        self.head_arc_feedforward = arc_feedforward or FeedForward(
            encoder_dim, 1, arc_representation_dim, Activation.by_name("elu")()
        )
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True
        )

        self.head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")()
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.decoder = decoder

        self.pos_tag_embedding = None
        if pos_embed_dim:
            self.pos_tag_embedding = Embedding(
                vocab.get_vocab_size("pos"), pos_embed_dim
            )

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)
        self._head_sentinel = torch.nn.Parameter(
            torch.randn([1, 1, encoder.get_output_dim()])
        )

        representation_dim = text_field_embedder.get_output_dim()
        if self.pos_tag_embedding is not None:
            representation_dim += self.pos_tag_embedding.get_output_dim()

        check_dimensions_match(
            representation_dim,
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )

        check_dimensions_match(
            tag_representation_dim,
            self.head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )
        check_dimensions_match(
            arc_representation_dim,
            self.head_arc_feedforward.get_output_dim(),
            "arc representation dim",
            "arc feedforward output dim",
        )

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        tags = self.vocab.get_token_to_index_vocabulary("pos")
        punctuation_tag_indices = {
            tag: index for tag, index in tags.items() if tag in POS_TO_IGNORE
        }
        self._pos_to_ignore = set(punctuation_tag_indices.values())
        logger.info(
            f"Found POS tags corresponding to the following punctuation : {punctuation_tag_indices}. "
            "Ignoring words with these POS tags for evaluation."
        )

        self._attachment_scores = AttachmentScores()
        initializer(self)

    def embeddings_from_tokens_and_pos(
        self, tokens: Dict[str, torch.Tensor], pos_tags: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:

        embedded_text_input = self.text_field_embedder(tokens)
        if pos_tags is not None and self.pos_tag_embedding is not None:
            embedded_pos_tags = self.pos_tag_embedding(pos_tags)
            embedded_text_input = torch.cat(
                [embedded_text_input, embedded_pos_tags], -1
            )
        elif self.pos_tag_embedding is not None:
            raise ConfigurationError(
                "Model uses a POS embedding, but no POS tags were passed."
            )

        mask = get_text_field_mask(tokens)
        return embedded_text_input, mask

    def encode_text(
        self, embedded_text_input: torch.Tensor, mask: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        # shape: (batch, seq_len, embed_dim)
        embedded_text_input = self._input_dropout(embedded_text_input)
        # shape: (batch, seq_len, encoder_dim)
        encoded_text = self.encoder(embedded_text_input, mask)

        batch_size, _, encoding_dim = encoded_text.size()
        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        # shape: (batch, seq_len + 1, encoder_dim)
        encoded_text = self._dropout(torch.cat([head_sentinel, encoded_text], 1))
        # shape: (batch, seq_len + 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        return encoded_text, mask

    def generate_head_tag_representations(
        self, encoded_text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: (batch, seq_len + 1, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        # shape: (batch, seq_len + 1, tag_representation_dim)
        child_tag_representation = self._dropout(
            self.child_tag_feedforward(encoded_text)
        )
        return head_tag_representation, child_tag_representation

    def generate_arc_scores(
        self, encoded_text: torch.Tensor, mask: torch.LongTensor
    ) -> torch.Tensor:
        float_mask = mask.float()

        # shape: (batch, seq_len + 1, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        # shape: (batch, seq_len + 1, arc_representation_dim)
        child_arc_representation = self._dropout(
            self.child_arc_feedforward(encoded_text)
        )

        # shape (batch, seq_len + 1, seq_len + 1)
        attended_arcs = self.arc_attention(
            head_arc_representation, child_arc_representation
        )

        minus_inf = -1e8
        minus_mask = (1 - float_mask) * minus_inf
        # shape (batch, seq_len + 1, seq_len + 1)
        attended_arcs = (
            attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)
        )
        return attended_arcs

    @overrides
    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.LongTensor],
        pos_tags: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, sequence_length)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        pos_tags : ``torch.LongTensor``, required
            The output of a ``SequenceLabelField`` containing POS tags.
            POS tags are required regardless of whether they are used in the model,
            because they are used to filter the evaluation metric to only consider
            heads of words which are not punctuation.
        metadata : List[Dict[str, Any]], optional (default=None)
            A dictionary of metadata for each batch element which has keys:
                words : ``List[str]``, required.
                    The tokens in the original sentence.
                pos : ``List[str]``, required.
                    The dependencies POS tags for each word.
        head_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape ``(batch_size, sequence_length)``.
        head_indices : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length)``.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        arc_loss : ``torch.FloatTensor``
            The loss contribution from the unlabeled arcs.
        loss : ``torch.FloatTensor``, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : ``torch.FloatTensor``
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : ``torch.FloatTensor``
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """

        embedded_text_input, mask = self.embeddings_from_tokens_and_pos(
            tokens, pos_tags
        )

        predicted_heads, predicted_head_tags, mask, loss = self._parse(
            embedded_text_input, mask, head_tags, head_indices
        )

        if head_indices is not None and head_tags is not None:
            evaluation_mask = self._get_mask_for_eval(mask[:, 1:], pos_tags)
            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self._attachment_scores(
                predicted_heads[:, 1:],
                predicted_head_tags[:, 1:],
                head_indices,
                head_tags,
                evaluation_mask,
            )

        output_dict = {
            "heads": predicted_heads,
            "head_tags": predicted_head_tags,
            "loss": loss,
            "mask": mask,
            "words": [meta["words"] for meta in metadata],
            "pos": [meta["pos"] for meta in metadata],
        }

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        head_tags = output_dict.pop("head_tags").cpu().detach().numpy()
        heads = output_dict.pop("heads").cpu().detach().numpy()
        mask = output_dict.pop("mask")
        lengths = get_lengths_from_binary_sequence_mask(mask)
        head_tag_labels = []
        head_indices = []
        for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
            instance_heads = list(instance_heads[1:length])
            instance_tags = instance_tags[1:length]
            labels = [
                self.vocab.get_token_from_index(label, "head_tags")
                for label in instance_tags
            ]
            head_tag_labels.append(labels)
            head_indices.append(instance_heads)

        output_dict["predicted_dependencies"] = head_tag_labels
        output_dict["predicted_heads"] = head_indices
        return output_dict

    def _parse(
        self,
        embedded_text_input: torch.Tensor,
        mask: torch.LongTensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: (batch, seq_len + 1, encoder_dim)
        encoded_text, mask = self.encode_text(embedded_text_input, mask)
        batch_size = encoded_text.size(0)
        # shape: (batch, seq_len + 1, tag_representation_dim),
        # shape: (batch, seq_len + 1, tag_representation_dim)
        (
            head_tag_representation,
            child_tag_representation,
        ) = self.generate_head_tag_representations(encoded_text)
        # shape: (batch, seq_len + 1, seq_len + 1)
        attended_arcs = self.generate_arc_scores(encoded_text, mask)

        if head_indices is not None:
            # shape: (batch, seq_len + 1)
            head_indices = torch.cat(
                [head_indices.new_zeros(batch_size, 1), head_indices], 1
            )
        if head_tags is not None:
            # shape: (batch, seq_len + 1)
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)

        decoding = (
            "greedy"
            if self.training or not self.use_mst_decoding_for_validation
            else "mst"
        )

        predicted_heads, predicted_head_tags = self.decoder.infer_tree(
            head_tag_representation=head_tag_representation,
            child_tag_representation=child_tag_representation,
            arc_scores=attended_arcs,
            mask=mask,
            decoding=decoding,
        )

        if head_tags is not None and head_indices is not None:
            nll = self.decoder(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                arc_scores=attended_arcs,
                mask=mask,
                head_indices=head_indices,
                head_tags=head_tags,
            )
        else:
            nll = self.decoder(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                arc_scores=attended_arcs,
                mask=mask,
                head_indices=predicted_heads,
                head_tags=predicted_head_tags,
            )
        return predicted_heads, predicted_head_tags, mask, nll

    def _get_mask_for_eval(
        self, mask: torch.LongTensor, pos_tags: torch.LongTensor
    ) -> torch.LongTensor:
        """
        Dependency evaluation excludes words are punctuation.
        Here, we create a new mask to exclude word indices which
        have a "punctuation-like" part of speech tag.

        Parameters
        ----------
        mask : ``torch.LongTensor``, required.
            The original mask.
        pos_tags : ``torch.LongTensor``, required.
            The pos tags for the sequence.

        Returns
        -------
        A new mask, where any indices equal to labels
        we should be ignoring are masked.
        """
        new_mask = mask.detach()
        for label in self._pos_to_ignore:
            label_mask = pos_tags.eq(label).long()
            new_mask = new_mask * (1 - label_mask)
        return new_mask

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._attachment_scores.get_metric(reset)
