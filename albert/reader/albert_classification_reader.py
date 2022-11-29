from __future__ import absolute_import

from typing import Dict, Iterable, List, Optional

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import LabelField, MetadataField, TextField
from allennlp.data.token_indexers import PretrainedBertIndexer


def convert_to_tokens(wordpieces: List[str]) -> List[Token]:
    word = wordpieces[0]
    words = []
    for idx in range(1, len(wordpieces)):
        if wordpieces[idx].startswith("##"):
            word += wordpieces[idx].lstrip("##")
        else:
            words.append(word)
            word = wordpieces[idx]
    if word != "":
        words.append(word)
    return [Token(word) for word in words]


@DatasetReader.register("albert_classification_reader")
class AlbertClassificationReader(DatasetReader):
    def __init__(
        self,
        bert_model: str,
        bert_index: str = "bert",
        use_starting_offsets: bool = False,
        do_lowercase: bool = False,
        label_namespace: str = "labels",
        never_lowercase: List[str] = None,
        max_pieces: int = 512,
        truncate_long_sequences: bool = True,
        lazy: bool = False,
    ) -> None:
        super(AlbertClassificationReader, self).__init__(lazy=lazy)
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=do_lowercase
        )
        self.token_indexers = {
            bert_index: PretrainedBertIndexer(
                pretrained_model=bert_model,
                use_starting_offsets=use_starting_offsets,
                do_lowercase=do_lowercase,
                never_lowercase=never_lowercase,
                max_pieces=max_pieces,
                truncate_long_sequences=truncate_long_sequences,
            )
        }
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as infile:
            for line in infile:
                line = line.strip()
                if line != "":
                    label, lang, review = line.split("\t", 2)
                    yield self.text_to_instance(
                        text=review, lang=lang, label=int(label)
                    )

    @overrides
    def text_to_instance(
        self,  # pylint: disable=arguments-differ
        text: str,
        lang: Optional[str] = None,
        label: Optional[int] = None,
    ) -> Instance:
        fields: Dict[str, Instance] = {}
        tokenized_string: List[str] = self.tokenizer.tokenize(text)
        tokens: List[Token] = convert_to_tokens(tokenized_string)
        fields["tokens"] = TextField(tokens, self.token_indexers)
        fields["metadata"] = MetadataField(
            {
                "words": [token.text for token in tokens],
                "tokenized": tokenized_string,
                "lang": lang,
            }
        )
        if label is not None:
            fields["label"] = LabelField(
                label=label, label_namespace=self.label_namespace, skip_indexing=True
            )
        return Instance(fields)
