import os
import sys
import time
import asyncio
import inspect
import subprocess

from functools import wraps
from types import MethodType
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Concatenate, Coroutine, Sequence, cast
from collections.abc import MutableMapping

from render_core import Box, Value, Fragment, to_str

from util import log, escape, html_escape, cleanup_text
from motto import hitokoto
from segments import (
    BaseElement,
    Segment,
    Element,
    Style,
    Link,
    Pre,
    BlockQuote,
    Raw,
    Code,
    Bold,
    Italic,
    Underline,
    Strikethrough,
    Spoiler,
)


async def _run[T](coro: Awaitable[T]) -> T | None:
    try:
        return await coro
    except Exception:
        log.exception('Promise: coroutine failed')


type Then[T, U] = Callable[[T], U | Coroutine[Any, Any, U]]


async def _chained(prev: Awaitable, callbacks: Sequence[Then]) -> Any:
    out = await prev
    for callback in callbacks:
        log.debug('then: got %r, calling %r', out, callback)
        out = callback(out)
        if isinstance(out, Promise):
            log.debug('then: awaiting task %r', out._task)
            out = await out._task
    log.debug('then: final result %r', out)
    return out


class Promise[T: Value | None](Box):
    __slots__ = ('_task',)

    def __init__(self, coro: Awaitable[T]):
        super().__init__()
        self._task = asyncio.create_task(_run(coro))

    def then(self, *callbacks: Then) -> Promise:
        # `callback` is expected to be a `SubDoc` with a `scope`, so we can call
        # it safely while not rendering.
        if not callbacks:
            return self
        if not all(callable(f) for f in callbacks):
            raise TypeError(f'Promise.then: callback must be callable, got {callbacks}')
        return Promise(_chained(self._task, callbacks))

    def __repr__(self) -> str:
        return f'<Promise task={self._task!r}>'


def to_segment(val: Value | None) -> Segment:
    if val is None:
        return ''

    if isinstance(val, BaseElement):
        return val

    if isinstance(val, Fragment):
        out = []
        parts = []
        for v in val:
            # Fragment is flattened when iterated, so returned seg cannot be
            # another Sequence.
            if s := cast(BaseElement | str, to_segment(v)):
                # Exclude empty strings and sequences.
                if isinstance(s, str):
                    # Join consecutive strings.
                    if s:
                        parts.append(s)
                else:
                    if parts:
                        out.append(''.join(parts))
                        parts.clear()
                    out.append(s)
        if parts:
            out.append(''.join(parts))

        if len(out) == 1:
            return out[0]
        return out or ''

    return to_str(val)


_funcs: dict[str, Callable] = {}
_methods: dict[str, Callable | None] = {}


def _inspect(func: Callable, name: str | None = None) -> tuple[str, bool]:
    if name is None:
        name = func.__name__.strip('_')

    sig = inspect.signature(func)
    is_method = 'self' in sig.parameters

    if name in _funcs or name in _methods:
        raise ValueError(f'{func} is already registered')

    return name, is_method


def public[**P, R](func: Callable[P, R], name: str | None = None) -> Callable[P, R]:
    name, is_method = _inspect(func, name=name)
    if is_method:
        _methods[name] = None
    else:
        _funcs[name] = func
    return func


def trusted[**P, R](
    func: Callable[Concatenate['Bridge', P], R] | Callable[P, R],
    *,
    name: str | None = None,
) -> Callable[Concatenate['Bridge', P], R]:
    name, is_method = _inspect(func, name=name)

    if is_method:
        func = cast(Callable[Concatenate['Bridge', P], R], func)

        @wraps(func)
        def wrapper(self: 'Bridge', *args: P.args, **kwargs: P.kwargs) -> R:
            if self._trusted is None:
                log.warning('Bridge: unauthorized access to %s', name)
                raise PermissionError('unauthorized')
            log.debug('Bridge: authorized %s for %s', self._trusted, name)
            return func(self, *args, **kwargs)

        _methods[name] = None
        return wrapper
    else:
        func = cast(Callable[P, R], func)

        @wraps(func)
        def wrapper2(self: 'Bridge', *args: P.args, **kwargs: P.kwargs) -> R:
            if self._trusted is None:
                log.warning('Bridge: unauthorized access to %s', name)
                raise PermissionError('unauthorized')
            log.debug('Bridge: authorized %s for %s', self._trusted, name)
            return func(*args, **kwargs)

        _methods[name] = wrapper2
        return wrapper2


class Bridge(Box):
    __slots__ = ('_ctx', '_update_text', '_trusted')

    def __init__(
        self,
        ctx: MutableMapping[str, Value],
        update_text: Callable[[Segment], Awaitable[Value | None]],
        trusted: int | None = None,
    ) -> None:
        super().__init__()
        self._ctx = ctx
        self._update_text = update_text
        self._trusted = trusted

    def __repr__(self) -> str:
        return f'Bridge({self._trusted})'

    def _get_func(self, name: str) -> Callable[..., Value | None] | None:
        if name.startswith('_') or name.endswith('_'):
            return None
        if (val := Bridge.__dict__.get(name)) is not None:
            return MethodType(val, self)
        return _funcs.get(name)

    def __getattr__(self, name: str) -> Any:
        if (val := _funcs.get(name)) is not None:
            return val
        raise AttributeError(name)

    @trusted
    def edit_message(self, text) -> Promise:
        log.debug('Bridge: edit_message: %r', text)
        return Promise(self._update_text(to_segment(text)))

    @public
    def dbg(self) -> str:
        return '\n'.join(f'{k}={v!r}' for k, v in self._ctx.items())


@trusted
def uname() -> str:
    r = os.uname()
    return f'{r.sysname} {r.nodename} {r.release} {r.version} {r.machine}'


@trusted
def version() -> str:
    return sys.version


@trusted
def read_file(path) -> str:
    if not isinstance(path, str):
        raise TypeError(f'path must be a str, got {path!r}')

    with open(path, encoding='utf-8') as fp:
        return fp.read()


@trusted
def write_file(path, text) -> None:
    if not isinstance(path, str):
        raise TypeError(f'path must be a str, got {path!r}')

    text = to_str(text)
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(text)


trusted(eval, name='evil')
trusted(repr)


@trusted
def system(cmd: str) -> str:
    result = subprocess.check_output(
        cmd,
        shell=True,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=0.1,
        env={'LANG': 'C', 'LC_ALL': 'C'},
    )
    return result.strip()


@trusted
def sleep(seconds: float) -> Promise[None]:
    return Promise(asyncio.sleep(seconds))


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class ProcessResult(Box):
    stdout: str
    stderr: str
    returncode: int | None
    elapsed: float

    def __str__(self) -> str:
        return self.stdout


async def _communicate(cmd: str, input: str | None) -> ProcessResult:
    fut = asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.PIPE if input else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    buf = input.encode('utf-8') if input else None

    t0 = time.perf_counter()
    proc = await fut
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(buf), timeout=10)
        returncode = proc.returncode
    except asyncio.TimeoutError:
        proc.terminate()
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            returncode = proc.returncode
            if not returncode:
                returncode = 137  # Killed
        else:
            returncode = proc.returncode
            if not returncode:
                returncode = 124  # Timed out

    elapsed = time.perf_counter() - t0
    stdout = stdout.decode(errors='replace').strip()
    stderr = stderr.decode(errors='replace').strip()
    return ProcessResult(stdout, stderr, returncode, elapsed)


@trusted
def communicate(cmd, input='') -> Promise[ProcessResult]:
    return Promise(_communicate(to_str(cmd), to_str(input)))


public(time.time, name='time')
public(time.perf_counter, name='perf')
public(hitokoto)


@public
def date() -> str:
    return datetime.now().isoformat()


@public
def today() -> str:
    return datetime.now().strftime('%c')


@public
def escape_(text) -> str:
    return escape(to_str(text))


@public
def html_escape_(text) -> str:
    return html_escape(to_str(text))


def create_style[**P, T: Element](
    text: Value | None,
    factory: Callable[Concatenate[Segment, P], T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T | str:
    return factory(seg, *args, **kwargs) if (seg := to_segment(text)) else ''


@public
def pre(text) -> Pre | str:
    return create_style(text, Pre)


@public
def quote(text, expandable=True) -> BlockQuote | str:
    return create_style(text, BlockQuote, bool(expandable))


@public
def link(text, url) -> Link | str:
    return create_style(text, Link, url)


@public
def code(text) -> Style | str:
    return create_style(text, Code)


@public
def bold(text) -> Style | str:
    return create_style(text, Bold)


@public
def italic(text) -> Style | str:
    return create_style(text, Italic)


@public
def uline(text) -> Style | str:
    return create_style(text, Underline)


@public
def strike(text) -> Style | str:
    return create_style(text, Strikethrough)


@public
def spoiler(text) -> Style | str:
    return create_style(text, Spoiler)


@public
def raw(text) -> Raw | str:
    text = to_str(text)
    return Raw(text) if text else ''


@public
def cleanup(text) -> str:
    return cleanup_text(to_str(text))


class Deferred(Box, BaseElement):
    __slots__ = ('_func',)

    def __init__(self, func: Callable[[], Value]):
        self._func = func

    @property
    def inner(self) -> Segment:  # type: ignore
        r = to_segment(self._func())
        log.debug('fc: got %r from %r', r, self._func)
        return r


@public
def fc(thunk: Callable) -> Deferred:
    '''A "functional component".
    Note that this may exceed the length limit in `Formatter`.
    '''
    return Deferred(thunk)


for name, func in _methods.items():
    if func is not None:
        setattr(Bridge, name, func)

del _methods
