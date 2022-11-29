from __future__ import absolute_import

from collections import MutableMapping
from typing import Dict, Iterable, Union

from allennlp.data import DatasetReader, Instance


def get_test_data(
    test_param: Union[str, MutableMapping], reader: DatasetReader
) -> Union[Iterable[Instance], Dict[str, Iterable[Instance]]]:
    if test_param is None:
        return None
    # Base case
    if isinstance(test_param, str):
        return reader.read(test_param)
    # Recursive case
    if isinstance(test_param, MutableMapping):
        test_data = {}
        for key, param in test_param.items():
            test_data[key] = get_test_data(param, reader)
        return test_data
    raise RuntimeError(
        f"test param can only be string or dict. Passed {type(test_param)}"
    )
