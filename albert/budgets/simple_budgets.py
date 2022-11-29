from __future__ import absolute_import

from typing import Optional

from overrides import overrides

from allennlp.common import Registrable
from allennlp.data import Instance


class Budget(Registrable):
    """Tracks different ways Active Learning budgets are specified. Child classes specify the
    specifics of the different budgets.

    """

    def __init__(
        self, max_tokens: Optional[int] = None, max_instances: Optional[int] = None
    ) -> None:
        if max_tokens is None and max_instances is None:
            raise RuntimeError("Need to specify either max_tokens or max_instances")
        self._max_tokens = max_tokens
        self._tokens_so_far = 0
        self._max_instances = max_instances
        self._instances_so_far = 0

    def get_max_instances(self) -> Optional[int]:
        return self._max_instances

    def get_max_tokens(self) -> Optional[int]:
        return self._max_tokens

    def can_add_instance(self, instance: Instance) -> bool:
        """This is overridden by children classes. This class
        checks if an instance can be added to the budget, based on
        the kind of budget it is, and updates local parameters if need be.
        """
        raise NotImplementedError("Child class implements this")

    def reset(self) -> None:
        """This resets the statistics of the budget"""
        raise NotImplementedError("Child class implements this")


@Budget.register("token")
class TokenBudget(Budget):
    """
    Tracks budget at a per token level.
    """

    def get_budget_size(self) -> int:
        return self._max_tokens

    def set_budget_size(self, size: int) -> None:
        self._max_tokens = size

    @overrides
    def reset(self) -> None:
        self._tokens_so_far = 0

    @overrides
    def can_add_instance(self, instance: Instance) -> bool:
        num_tokens = len(instance["tokens"].tokens)
        tokens_on_adding_instance = self._tokens_so_far + num_tokens
        if tokens_on_adding_instance <= self._max_tokens:
            # Can add instance
            self._tokens_so_far += num_tokens
        return tokens_on_adding_instance <= self._max_tokens


@Budget.register("instance")
class InstanceBudget(Budget):
    """
    Tracks budget at a per instance level.
    """

    def get_budget_size(self) -> int:
        return self._max_instances

    def set_budget_size(self, size: int) -> None:
        self._max_instances = size

    @overrides
    def reset(self) -> None:
        self._instances_so_far = 0

    @overrides
    def can_add_instance(self, instance: Instance) -> bool:
        # pylint: disable=unused-argument
        self._instances_so_far += 1
        return self._instances_so_far <= self._max_instances
