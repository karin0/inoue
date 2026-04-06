import os
import ast
import time
import logging
import functools

from datetime import datetime
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Callable, Literal, TypeGuard, overload

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


def date_func() -> str:
    return datetime.today().isoformat()


def today_func() -> str:
    return datetime.today().strftime('%c')


EVAL_FUNCS = {
    'time': time.time,
    'date': date_func,
    'today': today_func,
    'len': len,
}


class ScopeProxy:
    def __init__(self, ctx: 'ScopedContext', prefix: str):
        self._ctx = ctx
        self._prefix = prefix

    def __getattr__(self, name: str) -> Value:
        trace('eval: scope: %s %s', self._prefix, name)
        self._ctx._cb._consume_gas()
        return self._ctx._data[self._prefix + name]


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


class Context(MutableMapping[str, Value]):
    def setitem_with(self, key: str, val: Value, op: str) -> None:
        self[key] = val


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

    def set(self, name: str, val: Value):
        # `key, _ = self.resolve_raw(name)` ...?
        # We don't resolve the key here, since we always want to set
        # in the current scope.
        key = self._scopes[-1] + name
        self._data[key] = val
        trace('set: %s = %r', key, val)

    def set_with(self, name: str, val: Value, op: str):
        key = self._scopes[-1] + name
        self._data.setitem_with(key, val, op)
        trace('set_with: %s %s %r', key, op, val)

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

        r = func(*args, **kwargs)
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
