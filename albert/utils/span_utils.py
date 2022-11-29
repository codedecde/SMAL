from __future__ import absolute_import

import re
from typing import Dict, List, Optional, Set, Tuple, Union

from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
    bio_tags_to_spans,
    bioul_tags_to_spans,
    iob1_tags_to_spans,
    to_bioul,
)


def to_bio(tag_sequence: List[str]) -> List[str]:
    """This converts a IOB1 tagging sequence to a BIO format"""
    new_tag_sequece = ["O"] * len(tag_sequence)
    if tag_sequence:
        new_tag_sequece[0] = re.sub(r"^I-", "B-", tag_sequence[0])
        for index in range(1, len(tag_sequence)):
            prev_tag = re.sub(r"^.*-", "", tag_sequence[index - 1])
            curr_tag = re.sub(r"^.*-", "", tag_sequence[index])
            if prev_tag != curr_tag:
                new_tag_sequece[index] = re.sub(r"^I-", "B-", tag_sequence[index])
            else:
                new_tag_sequece[index] = tag_sequence[index]
    return new_tag_sequece


def resolve_entity_in_token_list(
    tokens: List[Token],
    entity: List[Token],
    ignore_case: bool = False,
    ignore_words: Optional[Set[str]] = None,
) -> List[Tuple[int, int]]:
    """Given a list of tokens ``tokens'' and an entity ``entity'', tries to find the positions of occurance
    of that entity in the text. Note that the entity positions are inclusive
    If ignore case is true, then the match is considered regardless of the casing of the tokens.
    Note that we can also resolve the entity if the it occurs in the tokens, with words in
    ``ignore_words`` interspersed. (Eg: this allows us to tag next week preferably wednesday,
    even when the entitiy is next week wednesday, if preferably is in ignore_words)
    """
    pos_list: List[Tuple[int, int]] = []
    pos: int = 0
    start: Optional[int] = None
    ignore_words: Set[str] = ignore_words or set()
    for index, token in enumerate(tokens):
        token_text = token.text.lower() if ignore_case else token.text
        entity_text = entity[pos].text.lower() if ignore_case else entity[pos].text
        if token_text == entity_text:
            if pos == 0:
                start = index
            pos += 1
        elif token_text in ignore_words:
            continue
        else:
            entity_text = entity[0].text.lower() if ignore_case else entity[0].text
            pos, start = (1, index) if entity_text == token_text else (0, None)
        if pos == len(entity):
            # we found an occurance
            pos_list.append((start, index))
            start = None
            pos = 0
    return pos_list


def get_span_words(
    tokens: List[Union[str, Token]], tags: List[str], encoding="IOB1"
) -> Dict[str, List[str]]:
    """Give a list of tokens, and a list of tags, we extract out the entities of different types,
    where each entity is represented as a string
    """

    def _convert_to_token_list(tokens: List[Union[str, Token]]) -> List[Token]:
        new_tokens = []
        for token in tokens:
            if isinstance(token, Token):
                new_tokens.append(token)
            elif isinstance(token, str):
                new_tokens.append(Token(token))
            else:
                raise RuntimeError(f"Type {type(token)} not recognized")
        return new_tokens

    tokens = _convert_to_token_list(tokens)
    span_function = {
        "iob1": iob1_tags_to_spans,
        "bio": bio_tags_to_spans,
        "bioul": bioul_tags_to_spans,
    }
    if encoding.lower() not in span_function:
        raise RuntimeError(f"The encoding {encoding} is not supported")

    spans = span_function[encoding.lower()](tags)
    retval = {}
    for label, (s_ix, e_ix) in spans:
        if label not in retval:
            retval[label] = []
        span_tokens = []
        for ix in range(s_ix, e_ix + 1):
            span_tokens.append(tokens[ix].text)
        retval[label].append(" ".join(span_tokens))
    return retval


def generate_tags(
    tokens: List[Token],
    entities: Dict[str, List[List[Token]]],
    encoding: str = "iob1",
    ignore_case: Union[bool, Dict[str, bool]] = False,
    ignore_words: Optional[Set[str]] = None,
) -> Tuple[List[str], Dict[str, List[List[Token]]]]:
    """Generates tags from entities.
    Parameters:
        tokens (List[str]): The tokens to tag
        entities (Dict[str, List[List[str]]]): The entities. Each key marks the tag type
            Each element in the List represents a tokenized entity
        encoding (str): The encoding type generated. We can create encodings in "IOB1", "BIO" and
            "BIOUL" formats
        ignore_case (Dict[str, bool]): Ignore casing while matching tokens for each tag type
        ignore_words (Set[str]): Ignore the words occuring in the tokens that exist here,
            while generating tags (Eg: this allows us to tag next week preferably wednesday,
            even when the entitiy is next week wednesday, if preferably is in ignore_words)
    Returns:
        tags (List[str]): The tags. len(tags) == len(tokens)
        unresolved (Dict[str, List[List[Token]]]): All the entities we could not resolve
    """
    if isinstance(ignore_case, bool):
        ignore_case = {x: ignore_case for x in entities}
    spans: List[Tuple[int, int, str]] = []
    unresolved: Dict[str, List[List[str]]] = {}
    for tag, entities in entities.items():
        for entity in entities:
            pos_list = resolve_entity_in_token_list(
                tokens, entity, ignore_case[tag], ignore_words
            )
            if pos_list:
                for start, end in pos_list:
                    spans.append((start, end, tag))
            else:
                if tag not in unresolved:
                    unresolved[tag] = []
                unresolved[tag].append(entity)
    if not spans:
        return (["O"] * len(tokens), unresolved)
    spans = sorted(spans)
    pruned_spans: List[Tuple[int, int, str]] = []
    # we prune entities that overlap. The longer entitiy is kept
    # Caveat: If both the entites are of the same tag, then
    # we simply collapse them into one big entity
    pruned_spans.append(spans[0])
    for six, eix, tag in spans[1:]:
        prev_six, prev_eix, prev_tag = pruned_spans[-1]
        if prev_eix < six:
            pruned_spans.append((six, eix, tag))
        else:
            # entities overlap
            if prev_tag == tag:
                # case 1: the tags for the previous and the
                # current entity are the same
                # We combine them into one big entity
                six = prev_six
                eix = max(eix, prev_eix)
                pruned_spans.pop()
                pruned_spans.append((six, eix, tag))
            else:
                # case 2: the tags don't match. In that case
                # we keep the larger one
                current_entity_len = eix - six + 1
                prev_entity_len = prev_eix - prev_six + 1
                if current_entity_len > prev_entity_len:
                    pruned_spans.pop()
                    pruned_spans.append((six, eix, tag))
    # we now assign the tags
    tags = ["O"] * len(tokens)
    # base case: the first entity
    six, eix, tag = pruned_spans[0]
    for index in range(six, eix + 1):
        tags[index] = f"I-{tag}"
    for span_index in range(1, len(pruned_spans)):
        n_six, n_eix, n_tag = pruned_spans[span_index]
        for index in range(n_six, n_eix + 1):
            if index == eix + 1 and n_tag == tag:
                tags[index] = f"B-{n_tag}"
            else:
                tags[index] = f"I-{n_tag}"
        six, eix, tag = n_six, n_eix, n_tag
    if encoding.lower() != "iob1":
        encoding_to_function = {"bioul": to_bioul, "bio": to_bio}
        if encoding.lower() not in encoding_to_function:
            raise RuntimeError(f"encoding: {encoding.lower()} not supported")
        tags = encoding_to_function[encoding.lower()](tags)
    return tags, unresolved
