from typing import Iterable
from collections import UserDict, OrderedDict
from collections.abc import MutableMapping

from simpleeval import simple_eval

# Allowed types for context values, as allowed by `simpleeval` by default (except None).
# Other types should have been disallowed in `simpleeval`.
type Value = str | int | float | bool | complex | bytes

LRU_CAPACITY = 128


class LRUDict(MutableMapping[str, Value]):
    def __init__(self):
        self._data = OrderedDict()

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Value:
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: str, value: Value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > LRU_CAPACITY:
            self._data.popitem(last=False)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


# Static context variables are persisted across different `RenderInterpreter` instances.
persisted = LRUDict()
PM_PREFIX = 'pm.'


def get_pm_key(key: str) -> str | None:
    if len(pm_key := key.removeprefix(PM_PREFIX)) != len(key):
        return pm_key


class OverriddenDict(UserDict):
    def __init__(self, overrides: dict[str, Value]):
        super().__init__()
        self.overrides = overrides

    def __getitem__(self, key: str) -> Value:
        if (v := self.overrides.get(key)) is not None:
            return v
        if pm_key := get_pm_key(key):
            return persisted[pm_key]
        return self.data[key]

    # For `get` method to work.
    def __contains__(self, key: str) -> bool:  # type: ignore[override]
        if key in self.overrides:
            return True
        if pm_key := get_pm_key(key):
            return pm_key in persisted
        return key in self.data

    def __setitem__(self, key: str, val: Value):
        # For `=` operator.
        if pm_key := get_pm_key(key):
            if key not in self.overrides:
                persisted[pm_key] = val
            return

        # We always set the value in the underlying dict, even if the value
        # is overridden, so the natural order of keys is preserved as defined by
        # documents and won't change after markup buttons are activated.
        self.data[key] = val

    def setdefault(self, key: str, default: Value) -> Value:  # type: ignore[override]
        # For `?=` operator.
        if pm_key := get_pm_key(key):
            if (r := self.overrides.get(key)) is not None:
                return r
            return persisted.setdefault(pm_key, default)

        # For same purpose as in `__setitem__`, we always set the value
        # in the underlying dict.
        # The static (persisted) keys are bypassed anyway.
        r0 = self.data.setdefault(key, default)
        if (r := self.overrides.get(key)) is not None:
            return r

        return r0

    def setdefault_override(self, key: str, value: Value) -> Value:
        # For `:=` operator.
        #
        # This only affects the `overrides` dict, which has higher priority than
        # the underlying dict by their *values*, but are placed after those touched
        # by `=` and `?=` in the natural order of *keys*.
        #
        # There is no need to write back the overridden *value* either here or
        # in `__setitem__`, since `finalize()` already reflects the values from
        # `overrides`.
        return self.overrides.setdefault(key, value)

    def clone(self) -> 'OverriddenDict':
        new = OverriddenDict(self.overrides.copy())
        new.data = self.data.copy()
        return new

    def finalize(self) -> Iterable[tuple[str, Value]]:
        # Keys from the underlying dict are yielded first, in their natural order.
        # Technically the side effects of this method shouldn't affect our behavior,
        # since existing overrides can never be removed or modified.
        self.data.update(self.overrides)
        return self.data.items()


def to_str(v: Value) -> str:
    if isinstance(v, bool):
        return '1' if v else '0'
    return str(v)
