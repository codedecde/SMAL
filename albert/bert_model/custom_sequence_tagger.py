from __future__ import absolute_import

from typing import Dict, Optional

import torch

import allennlp.nn.util as util
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.models.crf_tagger import CrfTagger
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator


@Model.register("custom_crf_tagger")
class CustomCrfTagger(CrfTagger):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        seq2vec_encoder: Seq2VecEncoder,
        label_namespace: str = "labels",
        feedforward: Optional[FeedForward] = None,
        label_encoding: Optional[str] = None,
        include_start_end_transitions: bool = True,
        constrain_crf_decoding: bool = None,
        calculate_span_f1: bool = None,
        dropout: Optional[float] = None,
        verbose_metrics: bool = False,
        num_labels: Optional[int] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        top_k: int = 1,
    ) -> None:
        super(CustomCrfTagger, self).__init__(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            encoder=encoder,
            label_namespace=label_namespace,
            feedforward=feedforward,
            label_encoding=label_encoding,
            include_start_end_transitions=include_start_end_transitions,
            constrain_crf_decoding=constrain_crf_decoding,
            calculate_span_f1=calculate_span_f1,
            num_tags=num_labels,
            dropout=dropout,
            verbose_metrics=verbose_metrics,
            initializer=initializer,
            regularizer=regularizer,
            top_k=top_k,
        )
        self._seq2vec_encoder = seq2vec_encoder
        self._label_namespace = self.label_namespace

    def sentence_representation(
        self,  # pylint: disable=unused-argument
        tokens: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        # (batch, seq_len, embed_dim)
        embedded_text_input = self.text_field_embedder(tokens)
        # (batch, seq_len)
        mask = util.get_text_field_mask(tokens)

        # (batch, seq_len, encoding_dim)
        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        # (batch, sent_embed_dim)
        sentence_representation = self._seq2vec_encoder(encoded_text, mask)

        return sentence_representation
