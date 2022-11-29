from __future__ import absolute_import

import itertools
import logging
from typing import Dict, Iterable, List, Optional

from overrides import overrides

from allennlp.common import Params
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import (
    Field,
    LabelField,
    MetadataField,
    SequenceLabelField,
    TextField,
)
from allennlp.data.token_indexers import TokenIndexer

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    return False


@DatasetReader.register("multilingual_albert_reader")
class MultilingualAlbertReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        coding_scheme: str = "IOB1",
        label_namespace: str = "labels",
        lang_namespace: str = "lang_tags",
        replace_num_token: Optional[str] = None,
        lazy: bool = False,
    ) -> None:
        super(MultilingualAlbertReader, self).__init__(lazy)
        self.token_indexers = token_indexers
        self.label_namespace = label_namespace
        self.coding_scheme = coding_scheme
        self.replace_num_token = replace_num_token
        self.lang_namespace = lang_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r", encoding="ISO-8859-1") as data_file:
            logger.info(f"Reading instances from lines in file at : {file_path}")
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                if not is_divider:
                    tokens: List[str] = []
                    tags: List[str] = []
                    lang_list: List[str] = []
                    for line in lines:
                        token, lang, tag = line.split(" ")
                        tag = tag.strip()
                        if self.replace_num_token and token.isdigit():
                            token = self.replace_num_token
                        tokens.append(token)
                        lang_list.append(lang)
                        tags.append(tag)
                    langs = list(set(lang_list))
                    langs_string = "\n".join(lines)
                    assert len(langs) == 1, (
                        "Cannot have multiple languages for the same instance "
                        f"found {langs_string}"
                    )
                    yield self.text_to_instance(tokens, langs[0], tags)

    @overrides
    def text_to_instance(
        self,  # pylint: disable=arguments-differ
        tokens: List[str],
        lang: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = [Token(x) for x in tokens]
        sequence = TextField(tokens, self.token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"lang": lang, "words": tokens})
        fields["langs"] = LabelField(label=lang, label_namespace=self.lang_namespace)
        if tags:
            fields["tags"] = SequenceLabelField(tags, sequence, self.label_namespace)
        return Instance(fields)
