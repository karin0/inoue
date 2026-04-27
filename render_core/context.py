import os
import ast
import time
import logging
import functools

from datetime import datetime
from abc import ABC, abstractmethod
from collections.abc import MutableMapping, Sequence
from typing import Any, Callable, Iterator, Literal, TypeGuard, Iterable, overload

from simpleeval import SimpleEval, DEFAULT_FUNCTIONS, DISALLOW_FUNCTIONS

from .tco import Tco, TCO

log = logging.getLogger(__name__)
is_tracing = os.environ.get('TRACE') == '1'
is_not_quiet = os.environ.get('TRACE_QUIET') != '1'
trace = log.debug if is_tracing else lambda *_: None


class Box:
    __slots__ = ()

    def __str__(self) -> str:
        return ''

    def __repr__(self) -> str:
        return 'Box(...)'

    # Pretend to be a string in `simpleeval` to keep us transparent.
    def __add__(self, other) -> str:
        return str(self) + to_str(other)

    def __radd__(self, other) -> str:
        return to_str(other) + str(self)


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


class Fragment(Box, Sequence[Value]):
    '''An immutable fragment of values.'''

    __slots__ = ('_inner', '_flattened')

    def __init__(self, inner: list[Value]):
        self._inner = inner
        self._flattened = False

    def __str__(self) -> str:
        return ''.join(to_str(x) for x in self._flatten())

    def __repr__(self) -> str:
        return f'Fragment({self._flatten()!r})'

    @overload
    def __getitem__(self, index: int) -> Value: ...

    @overload
    def __getitem__(self, index: slice) -> list[Value]: ...

    def __getitem__(self, index: int | slice) -> Value | list[Value]:
        return self._flatten()[index]

    def __len__(self) -> int:
        return len(self._flatten())

    def __iter__(self) -> Iterator[Value]:
        return iter(self._flatten())

    def _flatten_inner(self) -> Iterable[Value]:
        for x in self._inner:
            if isinstance(x, Fragment):
                yield from x._flatten_inner()
            else:
                yield x

    def _flatten_copy(self) -> list[Value]:
        # Join all consecutive strings.
        new = []
        parts = []
        for x in self._flatten_inner():
            if isinstance(x, str):
                if x:
                    parts.append(x)
            else:
                if parts:
                    new.append(''.join(parts))
                    parts.clear()
                new.append(x)

        if parts:
            new.append(''.join(parts))

        return new

    def _flatten(self) -> list[Value]:
        if not self._flattened:
            self._inner = self._flatten_copy()
            self._flattened = True
        return self._inner

    def _trim(self, keep_newline: bool = True) -> Value:
        '''Similar to `trim_output`.'''
        if not (items := self._flatten_copy()):
            return ''

        start = 0
        while start < len(items) and isinstance(x := items[start], str):
            if x.isspace():
                start += 1
            else:
                items[start] = x.lstrip()
                break

        end = len(items)
        newline = not keep_newline
        while end > start and isinstance(x := items[end - 1], str):
            if x.isspace():
                end -= 1
                newline = newline or '\n' in x
            else:
                items[end - 1] = t = x.rstrip()
                newline = newline or x.find('\n', len(t)) >= 0
                break

        if (out := items[start:end]) and newline and keep_newline:
            out.append('\n')

        if len(out) == 1:
            return out[0]
        return Fragment(out) if out else ''


def trim_output(s: str, keep_newline: bool = True) -> str:
    '''Trim leading and trailing whitespace, but may keep a line break if the trailing whitespaces contain any.'''
    if not keep_newline:
        return s.strip()

    r = s.lstrip()
    s = r.rstrip()
    if len(s) != len(r) and s and r.find('\n', len(s)) >= 0:
        return s + '\n'
    return s


class ScopeProxy:
    __slots__ = ('_ctx', '_prefix')

    def __init__(self, ctx: 'ScopedContext', prefix: str):
        self._ctx = ctx
        self._prefix = prefix

    def __getattr__(self, name: str) -> Value:
        trace('eval: scope: %s %s', self._prefix, name)
        self._ctx._cb._consume_gas()
        return self._ctx._data[self._prefix + name]


class ContextCallbacks(ABC):
    __slots__ = ()

    @abstractmethod
    def _error(self, msg: str):
        pass

    @abstractmethod
    def _consume_gas(self):
        pass


class Context(MutableMapping[str, Value]):
    __slots__ = ()

    def setitem_with(self, key: str, val: Value, op: str) -> None:
        self[key] = val


@functools.lru_cache
def parse_ast(expr: str):
    return SimpleEval.parse(expr)


class ScopedContext:
    __slots__ = (
        '_scopes',
        '_prefixes',
        '_data',
        '_cb',
        '_funcs',
        '_eval',
        '_eval_str',
        '_eval_node',
    )

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
            if isinstance(func, Box) and callable(func):
                # A SubDoc.
                return func(*args, **kwargs)
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
