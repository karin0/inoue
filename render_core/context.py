import os
import time
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeGuard
from collections import UserDict, OrderedDict
from collections.abc import ItemsView, MutableMapping

from simpleeval import SimpleEval, DEFAULT_FUNCTIONS

from util import log

is_tracing = os.environ.get('TRACE') == '1' and log.isEnabledFor(logging.DEBUG)
trace = log.debug if is_tracing else lambda *_: None

# Allowed types for context values, as allowed by `simpleeval` by default (except None).
# Other types should have been disallowed in `simpleeval`.
type Value = str | int | float | bool | complex | bytes


def is_value_type(v: Any) -> TypeGuard[Value]:
    return isinstance(v, (str, int, float, bool, complex, bytes))


def _to_str(v) -> str | None:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, complex)):
        return str(v)
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')


def to_str(v: Value) -> str:
    if (s := _to_str(v)) is None:
        raise TypeError(f'bad value type: {type(v)}: {v}')
    return s


def fix_to_str(val) -> str:
    if isinstance(val, (list, tuple)) and all(is_value_type(v) for v in val):
        log.warning('Sequence result: %r (%r)', val, type(val))
        return str(val)
    log.error('Unsafe result: %r (%r)', val, type(val))
    return ''


def try_to_str(val) -> str:
    if val is None:
        return ''
    if (s := _to_str(val)) is not None:
        return s
    return fix_to_str(val)


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

    # For `=` operator.
    def __setitem__(self, key: str, val: Value):
        if pm_key := get_pm_key(key):
            if key not in self.overrides:
                persisted[pm_key] = val
            return

        # We always set the value in the underlying dict, even if the value
        # is overridden, so the natural order of keys is preserved as defined by
        # documents and won't change after markup buttons are activated.
        self.data[key] = val

    # For `:=` operator.
    # This only affects the `overrides` dict, which has higher priority than
    # the underlying dict by their *values*, but are placed after those touched
    # by `=` and `?=` in the natural order of *keys*.
    # However, the existing `overrides` keys are frozen and can NEVER be modified
    # or removed, even by `:=` itself.
    def setdefault_override(self, key: str, value: Value) -> Value:
        return self.overrides.setdefault(key, value)

    def items(self) -> ItemsView[str, Value]:
        # Keys from the underlying dict are yielded first, in their natural order.
        # Technically the side effects of this method shouldn't affect our behavior,
        # since existing overrides can never be removed or modified.
        self.data.update(self.overrides)
        return self.data.items()

    def debug(self):
        for k, v in self.items():
            trace('  %s = %r', k, v)
        for k, v in persisted.items():
            trace('  (pm) %s = %r', k, v)


def date_func() -> str:
    return datetime.today().isoformat()


def today_func() -> str:
    return datetime.today().strftime('%c')


EVAL_FUNCS = {
    '__time__': time.time,
    '__date__': date_func,
    '__today__': today_func,
}


class ScopeProxy:
    def __init__(self, ctx: 'ScopedContext', prefix: str):
        self._ctx = ctx
        self._prefix = prefix

    def __getattr__(self, name: str) -> Value:
        trace('eval: scope: %s %s', self._prefix, name)
        self._ctx._cb._consume_gas()
        return self._ctx._data[self._prefix + name]


class EvalFunctions(UserDict):
    def __init__(self, funcs: dict[str, Callable], stat_func: Callable[[], Any]):
        super().__init__(DEFAULT_FUNCTIONS, **EVAL_FUNCS, **funcs)
        self.stat = stat_func

    def __getitem__(self, name: str) -> Callable:
        trace('eval: funcs: %s', name)
        self.stat()
        return self.data[name]

    # Called by `simpleeval` internally, without `stat` invocation.
    def values(self):
        return self.data.values()


class ContextCallbacks(ABC):
    @abstractmethod
    def _error(self, msg: str):
        pass

    @abstractmethod
    def _consume_gas(self):
        pass


class ScopedContext:
    def __init__(
        self,
        ctx: OverriddenDict,
        callbacks: ContextCallbacks,
        funcs: dict[str, Callable],
    ):
        self._scopes: list[str] = ['']
        self._prefixes = {}
        self._data = ctx
        self._cb = callbacks
        funcs.setdefault('prefix', self._prefix_func)
        funcs: EvalFunctions = EvalFunctions(funcs, callbacks._consume_gas)
        self._eval = SimpleEval(functions=funcs, names=self)
        self._funcs = funcs.data

    def push(self, name: str):
        new = self._scopes[-1] + name + '.'
        trace('Entering scope: %s', new)
        self._scopes.append(new)
        self._prefixes[name] = new

    def pop(self):
        last = self._scopes.pop()
        trace('Leaving scope: %s', last)

    def _prefix_func(self) -> str:
        return self._scopes[-1]

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
                self._cb._error('undefined: ' + key)
            # `as_str` is ignored when `default` is used.
            return key, default

        trace('Got var: %s = %r', key, val)
        return key, to_str(val) if as_str else val

    def get(
        self, name: str, *, as_str: bool = False, allow_undef: bool = False
    ) -> Value:
        if allow_undef:
            _, val = self.resolve_raw(name, None, as_str=as_str)
            return '' if val is None else val
        _, val = self.resolve_raw(name, '', as_str=as_str)
        assert val is not None
        return val

    def current(self) -> str:
        return self._scopes[-1]

    def current_key(self, name: str) -> str:
        return self._scopes[-1] + name

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
    def __getitem__(self, name: str) -> Value | ScopeProxy | None:
        key, val = self.resolve_raw(name)
        trace('eval: item: %s -> %r', key, val)
        self._cb._consume_gas()
        if val is not None:
            return val

        if (prefix := self._prefixes.get(name)) is not None:
            return ScopeProxy(self, prefix)

        if (func := self._funcs.get(name)) is not None:
            # Call the function without arguments implicitly.
            r = func()
            if r is None or is_value_type(r):
                trace('Implicit function result: %s() -> %r', name, r)
                return r
            log.error('Unsafe function result: %s() -> %r (%r)', name, r, type(r))
            return None

        raise KeyError(key)

    def eval(self, expr: str) -> Value:
        val = self._eval.eval(expr)
        trace('eval: %s -> %s', expr, val)

        # Ensure a `Value`.
        if val is None:
            return ''

        if is_value_type(val):
            return val

        return fix_to_str(val)
