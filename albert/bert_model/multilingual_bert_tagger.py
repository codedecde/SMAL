from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel
from torch.nn.modules.linear import Linear

from albert.nn.util import convert_to_one_hot, select_subset
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.modules.token_embedders.bert_token_embedder import (
    BertEmbedder,
    PretrainedBertModel,
)
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, Metric, SpanBasedF1Measure


@Model.register("multilingual_bert_for_tagging")
class MultilingualBertForTagging(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, BertModel],
        top_layer_only: bool = True,
        dropout: float = 0.0,
        num_labels: int = None,
        num_langs: int = None,
        pooling: str = "cls",
        index: str = "bert",
        calculate_span_f1: bool = None,
        label_namespace: str = "labels",
        lang_namespace: str = "lang_tags",
        verbose_metrics: bool = False,
        label_encoding: Optional[str] = None,
        trainable: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(MultilingualBertForTagging, self).__init__(vocab, regularizer)
        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        self.bert_embedder = BertEmbedder(
            bert_model=self.bert_model, top_layer_only=top_layer_only
        )

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        in_features = self.bert_embedder.get_output_dim()

        self._lang_namespace = lang_namespace
        num_langs = num_langs or vocab.get_vocab_size(namespace=self._lang_namespace)
        self.num_langs = num_langs

        self._label_namespace = label_namespace
        num_labels = num_labels or vocab.get_vocab_size(namespace=self._label_namespace)
        self.num_labels = num_labels

        self._dropout = torch.nn.Dropout(p=dropout)
        self.tag_projection_layers = torch.nn.ModuleList(
            [
                TimeDistributed(Linear(in_features, num_labels))
                for _ in range(self.num_langs)
            ]
        )

        assert pooling in ["cls", "mean"]
        self._pooling = pooling

        self._index = index
        # Metrics
        self._verbose_metrics = verbose_metrics
        if calculate_span_f1 and not label_encoding:
            raise ConfigurationError(
                "calculate_span_f1 is True, but " "no label_encoding was specified."
            )
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3),
        }

        if calculate_span_f1 or label_encoding:
            self._f1_metric = SpanBasedF1Measure(
                vocab, tag_namespace=label_namespace, label_encoding=label_encoding
            )
        else:
            self._f1_metric = None

        initializer(self)

    def _get_cls_from_input_ids(
        self, input_ids: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        input_mask = (input_ids != 0).long()
        _, pooled = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=input_mask,
        )
        return pooled

    def sentence_representation(
        self, tokens: Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Parameters:
            tokens (Dict[str, torch.Tensor]): The tokens of the form batch x num_word-pieces x ..
        Returns:
            representation (torch.Tensor): A sentence representation. We currently support using the
                cls token and mean-pooling the sentence embeddings
        """
        # shape: (batch, num-wordpieces)
        input_ids = tokens[self._index]
        # shape: (batch, num-wordpieces)
        token_type_ids = tokens[f"{self._index}-type-ids"]
        representation = None
        if self._pooling == "cls":
            representation = self._get_cls_from_input_ids(input_ids, token_type_ids)
        elif self._pooling == "mean":
            # shape: (batch, num_tokens)
            offsets = tokens[f"{self._index}-offsets"]
            # shape: (batch, num_tokens, embed_size)
            embeddings = self.bert_embedder(input_ids, offsets, token_type_ids)
            # shape: (batch, embed_size)
            representation = embeddings.mean(1)
        else:
            raise NotImplementedError(f"{self._pooling} not supported.")
        # shape: (batch, cls_embed_dim)
        return representation

    @overrides
    def forward(
        self,  # pylint: disable=arguments-differ
        tokens: Dict[str, torch.Tensor],
        langs: torch.LongTensor,
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        # shape: (batch, num-wordpieces)
        input_ids = tokens[self._index]
        # shape: (batch, num-wordpieces)
        token_type_ids = tokens[f"{self._index}-type-ids"]
        # shape: (batch, num-tokens)
        offsets = tokens[f"{self._index}-offsets"]
        # shape: (batch, num-tokens)
        input_mask = tokens["mask"]

        batch_size, sequence_length = input_mask.size()

        # shape: (batch, num-tokens, embed-size)
        embeddings = self.bert_embedder(input_ids, offsets, token_type_ids)
        embeddings = self._dropout(embeddings)

        logits_list = []
        class_probabilities_list = []
        for tag_projection_layer in self.tag_projection_layers:
            # shape: (batch, num-tokens, num-tags)
            logits = tag_projection_layer(embeddings)
            logits_list.append(logits)
            # shape: (batch * num-tokens, num-tags)
            reshaped_log_probs = logits.view(-1, self.num_labels)
            # shape: (batch, num-tokens, num-tags)
            class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
                [batch_size, sequence_length, self.num_labels]
            )
            class_probabilities_list.append(class_probabilities)
        lang_one_hot = convert_to_one_hot(langs, self.num_langs)

        output_dict = {
            "logits": logits_list,
            "class_probabilities": class_probabilities_list,
            "mask": input_mask,
            "lang_mask": lang_one_hot,
        }

        if tags is not None:

            total_loss = 0
            for index in range(self.num_langs):
                # shape: (batch,)
                mask = lang_one_hot[:, index]
                num_examples = mask.sum()
                if num_examples == 0:
                    # We have no examples of this
                    # language in this minibatch
                    # continue on
                    continue
                logits = logits_list[index]
                # Select the mini-minibatch to work on
                minibatch_logits, minibatch_tags, minibatch_mask = select_subset(
                    mask, logits, tags, input_mask
                )
                # Handle BERT truncation
                if minibatch_tags.size(1) != minibatch_logits.size(1):
                    # Since bert can potentially have truncations, handle that
                    minibatch_tags = minibatch_tags[
                        :, : minibatch_logits.size(1)
                    ].contiguous()
                # shape: (1,)
                loss = sequence_cross_entropy_with_logits(
                    minibatch_logits, minibatch_tags, minibatch_mask
                )
                total_loss += loss

                for metric in self.metrics.values():
                    metric(minibatch_logits, minibatch_tags, minibatch_mask.float())
                if self._f1_metric is not None:
                    self._f1_metric(
                        minibatch_logits, minibatch_tags, minibatch_mask.float()
                    )

            output_dict["loss"] = total_loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        output_dict["tags"] = {}
        for index, all_predictions in enumerate(output_dict["class_probabilities"]):
            lang = self.vocab.get_token_from_index(
                index=index, namespace=self._lang_namespace
            )

            all_predictions, mask = Metric.unwrap_to_tensors(
                all_predictions, output_dict["mask"]
            )
            all_predictions = all_predictions.numpy()
            mask = mask.numpy()
            if all_predictions.ndim == 3:
                predictions_list = [
                    all_predictions[i] for i in range(all_predictions.shape[0])
                ]
                num_tokens: List[int] = [mask[i].sum() for i in range(mask.shape[0])]
            else:
                predictions_list = [all_predictions]
                num_tokens = [mask.sum()]
            all_tags = []
            for predictions, seq_len in zip(predictions_list, num_tokens):
                argmax_indices = np.argmax(predictions, axis=-1)
                tags = [
                    self.vocab.get_token_from_index(x, namespace="labels")
                    for x in argmax_indices
                ][:seq_len]
                all_tags.append(tags)
            output_dict["tags"][lang] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }

        if self._f1_metric is not None:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update(
                    {x: y for x, y in f1_dict.items() if "overall" in x}
                )
        return metrics_to_return
