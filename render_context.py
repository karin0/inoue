import os
import logging
from typing import Any, Callable, Iterable
from collections import UserDict, OrderedDict
from collections.abc import MutableMapping

from simpleeval import simple_eval

from util import log

is_tracing = os.environ.get('TRACE') == '1' and log.isEnabledFor(logging.DEBUG)
trace = log.debug if is_tracing else lambda *_: None

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
        trace('PM set: %s = %r', key, value)
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

    def clear(self) -> None:
        return self._data.clear()

    def items(self):
        return self._data.items()


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

    # For `set_or_del_raw` to work.
    def __delitem__(self, key: str):
        if pm_key := get_pm_key(key):
            if key not in self.overrides:
                del persisted[pm_key]
            return

        if key not in self.overrides:
            del self.data[key]

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

    def finalize(self) -> Iterable[tuple[str, Value]]:
        # Keys from the underlying dict are yielded first, in their natural order.
        # Technically the side effects of this method shouldn't affect our behavior,
        # since existing overrides can never be removed or modified.
        self.data.update(self.overrides)
        return self.data.items()

    def debug(self):
        for k, v in self.finalize():
            trace('  %s = %r', k, v)
        for k, v in persisted.items():
            trace('  (pm) %s = %r', k, v)


def to_str(v: Value) -> str:
    if isinstance(v, bool):
        return '1' if v else '0'
    return str(v)


class ScopeProxy:
    def __init__(self, ctx: OverriddenDict, prefix: str):
        self._data = ctx
        self._prefix = prefix

    def __getattr__(self, name: str) -> Value:
        r = self._data[self._prefix + name]
        trace('Scope proxy got: %s + %s = %r', self._prefix, name, r)
        return r


class ScopedContext:
    def __init__(self, ctx: OverriddenDict, error_func: Callable[[str], None]):
        self._scopes: list[str] = ['']
        self._prefixes = {'root': ''}
        self._data = ctx
        self._error = error_func

    def push(self, name: str):
        new = self._scopes[-1] + name + '.'
        trace('Entering scope: %s', new)
        self._scopes.append(new)
        self._prefixes[name] = new

    def pop(self):
        last = self._scopes.pop()
        trace('Leaving scope: %s', last)

    def resolve_raw(
        self, name: str, default: Value | None = None, as_str: bool = False
    ) -> tuple[str, Value | None]:
        for scope in reversed(self._scopes):
            key = scope + name
            if (val := self._data.get(key)) is not None:
                break
            trace('Var not found in scope: %s', key)
        else:
            key = self._scopes[-1] + name
            if default is not None:
                self._error('undefined: ' + key)
            # `as_str` is ignored when `default` is used.
            return key, default

        trace('Got var: %s = %r', key, val)
        return key, to_str(val) if as_str else val

    def get(self, name: str, *, as_str: bool = False) -> Value:
        _, val = self.resolve_raw(name, '', as_str=as_str)
        assert val is not None
        return val

    def set(self, name: str, val: Value, setter: Callable[[str, Value], Any]):
        # `key, _ = self.resolve_raw(name)` ...?
        # We don't resolve the key here, since we always want to set
        # in the current scope.
        key = self._scopes[-1] + name
        r = setter(key, val)
        trace('Assigning: %s = %r -> %r (%s)', key, val, r, setter.__name__)

    def set_or_del_raw(self, key: str, val: Value | None):
        trace('Set or del: %s = %r', key, val)
        if val is None:
            self._data.pop(key, None)
        else:
            self._data[key] = val

    # For `simpleeval` usage.
    def __getitem__(self, name: str) -> Value | ScopeProxy:
        if (prefix := self._prefixes.get(name)) is not None:
            return ScopeProxy(self._data, prefix)

        key, val = self.resolve_raw(name)
        trace('Getting item: %s -> %r', key, val)
        if val is None:
            raise KeyError((key, name))

        return val

    def eval(self, expr: str) -> Value:
        val = simple_eval(expr, names=self)
        trace('Evaluated: %s -> %s', expr, val)

        # Ensure a `Value`.
        if val is None:
            return ''

        return val
