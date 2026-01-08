import os
import ast
import time
import logging
import inspect
import functools

from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Literal, TypeGuard, Mapping, overload
from collections import UserDict, OrderedDict
from collections.abc import ItemsView, MutableMapping

from simpleeval import SimpleEval, DEFAULT_FUNCTIONS, DISALLOW_FUNCTIONS

from .tco import Tco, TCO

log = logging.getLogger(__name__)
is_tracing = os.environ.get('TRACE') == '1'
is_not_quiet = os.environ.get('TRACE_QUIET') != '1'
trace = log.debug if is_tracing else lambda *_: None


class Box:
    def __str__(self) -> str:
        return ''

    def __repr__(self) -> str:
        return 'Box(...)'


# Allowed types for context values, as allowed by `simpleeval` by default (except None).
# Other types should have been disallowed in `simpleeval`.
type Value = str | int | float | bool | complex | bytes | Box
type Items = ItemsView[str, Value]


def is_value_type(v: Any) -> TypeGuard[Value]:
    # bool is subclass of int
    return isinstance(v, (str, int, float, complex, bytes, Box))


def _to_str(v) -> str | None:
    if isinstance(v, str):
        return v
    if isinstance(v, bool):  # Must before `int` check!
        return '1' if v else '0'
    if isinstance(v, (int, float, complex, Box)):
        return str(v)
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


def try_to_str(val: Any) -> str:
    if val is None:
        return ''
    if (s := _to_str(val)) is not None:
        return s
    return fix_to_str(val)


def try_to_value(val: Any) -> Value:
    if val is None:
        return ''
    if is_value_type(val):
        return val
    return fix_to_str(val)


def try_to_value_or_none(val: Any) -> Value | None:
    if val is None or is_value_type(val):
        return val
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


def register_pm(storage: MutableMapping[str, Value]) -> None:
    global persisted
    persisted = storage


def get_pm_key(key: str) -> str | None:
    if len(pm_key := key.removeprefix(PM_PREFIX)) != len(key):
        return pm_key


class OverriddenDict(UserDict):
    def __init__(self, data: Mapping[str, Value], overrides: dict[str, Value]):
        for val in data.values():
            if not is_value_type(val):
                raise TypeError(f'bad data value type: {type(val)}: {val}')
        super().__init__(data)

        for val in overrides.values():
            if not is_value_type(val):
                raise TypeError(f'bad override value type: {type(val)}: {val}')
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

    def _compact(self):
        return dict(self.data, **self.overrides)

    def __iter__(self):
        # Keys from the underlying dict are yielded first, in their natural order.
        # Technically the side effects of this method shouldn't affect our behavior,
        # since existing overrides can never be removed or modified.
        return iter(self._compact())

    def __len__(self) -> int:
        return len(self._compact())

    def items(self) -> Items:
        return self._compact().items()

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
    'time': time.time,
    'date': date_func,
    'today': today_func,
}


class ScopeProxy:
    def __init__(self, ctx: 'ScopedContext', prefix: str):
        self._ctx = ctx
        self._prefix = prefix

    def __getattr__(self, name: str) -> Value:
        trace('eval: scope: %s %s', self._prefix, name)
        self._ctx._cb._consume_gas()
        return self._ctx._data[self._prefix + name]


type Context = MutableMapping[str, Value]


def call_with_context(
    func: Callable, ctx: Context, args: Iterable, kwargs: dict[str, Any]
) -> Any:
    try:
        sig = inspect.signature(func)
    except ValueError as e:
        # int, str
        trace('eval: no sig: %s: %s', func, e)
        return func(*args, **kwargs)

    for name, param in sig.parameters.items():
        if param.annotation is Context:
            break
    else:
        trace('eval: plain: %s %s', func, sig)
        return func(*args, **kwargs)

    trace('eval: inject: %s: %s (%s)', func, sig, name)

    params = tuple(p for p in sig.parameters.values() if p is not param)
    bound = inspect.Signature(params).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    arguments[name] = ctx
    real = sig.bind_partial()
    real.arguments = arguments
    trace('eval: bound: %s', real)
    return func(*real.args, **real.kwargs)


class ContextCallbacks(ABC):
    @abstractmethod
    def _error(self, msg: str):
        pass

    @abstractmethod
    def _consume_gas(self):
        pass

    @abstractmethod
    def _call_box(self, data, *args, **kwargs) -> Value | None:
        pass


@functools.lru_cache
def parse_ast(expr: str):
    return SimpleEval.parse(expr)


class ScopedContext:
    def __init__(
        self,
        ctx: Context,
        callbacks: ContextCallbacks,
        funcs: dict[str, Callable],
    ):
        self._scopes: list[str] = ['']
        self._prefixes = {}
        self._data = ctx
        self._cb = callbacks
        self._funcs = dict(
            DEFAULT_FUNCTIONS, **EVAL_FUNCS, prefix=self._prefix_func, **funcs
        )
        self._eval = SimpleEval(names=self)
        self._eval_str = self._eval.eval
        self._eval_node = self._eval._eval
        assert self._eval.nodes
        self._eval.nodes[ast.Call] = self._eval_call
        self._eval.nodes[ast.Subscript] = self._eval_subscript

    def push(self, name: str):
        new = self._scopes[-1] + name + '.'
        trace('Entering scope: %s', new)
        self._scopes.append(new)
        self._prefixes[name] = new

    def push_raw(self, prefix: str):
        trace('Entering raw scope: %s', prefix)
        self._scopes.append(prefix)

    def pop(self) -> str:
        last = self._scopes.pop()
        trace('Leaving scope: %s', last)
        return last

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

    def _prefix_func(self) -> str:
        return self._scopes[-1]

    def current(self) -> str:
        return self._scopes[-1]

    def last(self) -> str | None:
        return self._scopes[-2] if len(self._scopes) >= 2 else None

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
        trace('set_or_del_raw: %s = %r', key, val)
        if val is None:
            self._data.pop(key, None)
        else:
            self._data[key] = val

    # For `simpleeval` usage.
    def __getitem__(self, name: str) -> Value | ScopeProxy | None:
        trace('eval: get: %s', name)
        self._cb._consume_gas()
        if (r := self._get_eval_name(name)) is Tco:
            raise KeyError(name)
        return r

    def _get_eval_name(self, name: str) -> Value | ScopeProxy | None | TCO:
        key, val = self.resolve_raw(name)
        trace('eval: item: %s -> %r', key, val)

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

        return Tco

    def _eval_call(self, node: ast.Call):
        if is_tracing:
            trace('_eval_call: %s', ast.dump(node))
        self._cb._consume_gas()
        args = (self._eval_node(a) for a in node.args)
        kwargs = dict(self._eval_node(k) for k in node.keywords)

        func_node = node.func
        if isinstance(func_node, ast.Attribute):
            func = self._eval_node(func_node)
        elif isinstance(func_node, ast.Name):
            name = func_node.id
            func = self.get(name, allow_undef=True)
            if isinstance(func, Box):
                return self._cb._call_box(func, *args, **kwargs)
            func = self._funcs[name]
        else:
            raise NotImplementedError('unsupported call: ' + str(type(func_node)))

        if func in DISALLOW_FUNCTIONS:
            raise PermissionError('disallowed function')

        r = call_with_context(func, self._data, args, kwargs)
        if is_tracing:
            trace('_eval_call: %s -> %r', ast.dump(node), r)

        if r is None or is_value_type(r):
            return r

        if isinstance(r, list):  # str.split()
            for i, v in enumerate(r):
                r[i] = try_to_value_or_none(v)
            return r

        if isinstance(r, tuple):
            return tuple(try_to_value_or_none(v) for v in r)

        if isinstance(r, dict):
            return {
                try_to_value_or_none(k): try_to_value_or_none(v) for k, v in r.items()
            }

        return fix_to_str(r)

    def _eval_subscript(self, node: ast.Subscript):
        if is_tracing:
            trace('_eval_subscript: %s', ast.dump(node))
        self._cb._consume_gas()

        container_node = node.value
        if isinstance(container_node, ast.Name):
            container = self._get_eval_name(container_node.id)
            if container is Tco:
                slice = self._eval_node(node.slice)
                key = container_node.id + '.' + str(slice)
                raw_key, val = self.resolve_raw(key)
                trace('_eval_subscript: %s -> %s = %r', key, raw_key, val)
                if val is None:
                    raise KeyError(key)
                return val
        else:
            container = self._eval_node(container_node)

        slice = self._eval_node(node.slice)
        r = container[slice]  # type: ignore[index]
        trace('_eval_subscript: %r[%r] = %r', container, slice, r)
        return r

    @overload
    def eval(
        self, expr: str, *, allow_tco: Literal[True]
    ) -> Value | tuple[Any, tuple, dict]: ...

    @overload
    def eval(self, expr: str, *, allow_tco: Literal[False] = False) -> Value: ...

    def eval(
        self, expr: str, *, allow_tco: bool = False
    ) -> Value | tuple[Box, tuple, dict]:
        self._cb._consume_gas()
        tree = parse_ast(expr)

        # Manual TCO.
        if allow_tco:
            node = tree
            if isinstance(node, ast.Expr):
                node = node.value
            if isinstance(node, ast.Call) and isinstance(func := node.func, ast.Name):
                func = func.id
                key, val = self.resolve_raw(func)
                if isinstance(val, Box):
                    if is_tracing:
                        trace(
                            'eval: TCO: %s -> %s\n  %s = %r',
                            expr,
                            ast.dump(node),
                            key,
                            val,
                        )
                    args = tuple(self._eval_node(a) for a in node.args)
                    kwargs = dict(self._eval_node(k) for k in node.keywords)
                    return val, args, kwargs

        if is_tracing:
            trace('eval: ast: %s -> %s', expr, ast.dump(tree))

        val = self._eval_str(expr, tree)
        trace('eval: %s -> %s', expr, val)
        return try_to_value(val)
