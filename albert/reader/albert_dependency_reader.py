import logging
from typing import Dict, List, Tuple

from conllu import parse_incr
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.universal_dependencies import (
    UniversalDependenciesDatasetReader,
)
from allennlp.data.fields import Field, MetadataField, SequenceLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("albert_dependency_reader")
class AlbertDependencyReader(UniversalDependenciesDatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : ``Tokenizer``, optional, default = None
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        lazy: bool = False,
    ) -> None:
        super(AlbertDependencyReader, self).__init__(
            token_indexers=token_indexers,
            tokenizer=tokenizer,
            lazy=lazy,
            use_language_specific_pos=False,
        )

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info(f"Reading UD instances from conllu dataset at: {file_path}")

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if isinstance(x["id"], int)]

                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                tokens = [x["form"] for x in annotation]
                pos_tags = [x["upostag"] for x in annotation]
                lang_set = set(x["misc"]["lang"] for x in annotation)
                assert len(lang_set) == 1
                lang = list(lang_set)[0]
                yield self.text_to_instance(
                    tokens=tokens,
                    upos_tags=pos_tags,
                    lang=lang,
                    dependencies=list(zip(tags, heads)),
                )

    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokens: List[str],
        upos_tags: List[str],
        lang: str,
        dependencies: List[Tuple[str, int]] = None,
    ) -> Instance:
        """
        Parameters
        ----------
        tokens : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        lang: str, required.
            The language for this instance
        dependencies : ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(tokens))
        else:
            tokens = [Token(t) for t in tokens]

        text_field = TextField(tokens, self._token_indexers)
        fields["tokens"] = text_field
        fields["pos_tags"] = SequenceLabelField(
            upos_tags, text_field, label_namespace="pos"
        )
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [x[0] for x in dependencies], text_field, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [x[1] for x in dependencies],
                text_field,
                label_namespace="head_index_tags",
            )

        fields["metadata"] = MetadataField(
            {"words": tokens, "pos": upos_tags, "lang": lang}
        )
        return Instance(fields)
