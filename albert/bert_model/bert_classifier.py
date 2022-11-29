from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Union

import torch
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel

import allennlp.nn.util as nn_util
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.models.bert_for_classification import BertForClassification
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator


@Model.register("bert_classifier")
class BertClassifier(BertForClassification):
    """A wrapper over :class:``BertForClassification``, provided by AllenNLP.
    The original model does not allow for metadata to be passed. This wrappers allows
    you to pass the metadata.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, BertModel],
        dropout: float = 0.0,
        num_labels: int = None,
        index: str = "bert",
        label_namespace: str = "labels",
        trainable: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:

        super(BertClassifier, self).__init__(
            vocab=vocab,
            bert_model=bert_model,
            dropout=dropout,
            num_labels=num_labels,
            index=index,
            label_namespace=label_namespace,
            trainable=trainable,
            initializer=initializer,
            regularizer=regularizer,
        )

    def forward(  # type: ignore
        self,
        tokens: Dict[str, torch.LongTensor],
        label: Optional[torch.IntTensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        output_dict = super().forward(tokens=tokens, label=label)
        output_dict["metadata"] = metadata or []
        return output_dict

    def get_document_embedding_from_instances(
        self, instances: List[Instance]
    ) -> torch.Tensor:
        """Generate document embeddings for instances. Assumes the instance has the ``tokens`` key,
        which should be a TextField.

        Parameters
        ----------
        instances : List[Instance]
            The instance to generate embeddings for

        Returns
        -------
        torch.Tensor
            The document embedding. shape: (len(instances), bert_embed_dim)

        """
        batch = Batch(instances)
        batch.index_instances(self.vocab)
        tensor_dict = batch.as_tensor_dict()
        tensor_dict = nn_util.move_to_device(tensor_dict, self._get_prediction_device())
        return self.get_document_embedding(**tensor_dict)

    def get_document_embedding(
        self,
        tokens: Dict[str, torch.LongTensor],
        label: Optional[torch.IntTensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        # pylint: disable=unused-argument
        """Generate the document embeddings that were used for classification
        purposes

        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        label : Optional[torch.IntTensor], optional (default = None)
            From a ``LabelField``
        metadata : Optional[List[Dict[str, Any]]], optional (default = None)
            The metadata, usually the tokenized documents

        Returns
        -------
        torch.Tensor
            The document embedding. shape: (batch, bert_embed_dim)
        """
        with torch.no_grad():
            input_ids = tokens[self._index]
            token_type_ids = tokens[f"{self._index}-type-ids"]
            input_mask = (input_ids != 0).long()

            _, pooled = self.bert_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=input_mask,
            )
        return pooled
