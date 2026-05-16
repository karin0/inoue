import os
import sys
import time
import base64
import asyncio
import inspect
import weakref
import tempfile
import subprocess

from functools import wraps
from types import MethodType
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Concatenate,
    Coroutine,
    Sequence,
    Protocol,
    cast,
)
from collections.abc import MutableMapping

from render_core import Box, Value, Fragment, to_str

from util import (
    log,
    get_context,
    create_task,
    escape,
    html_escape,
    cleanup_text,
    cleanup_text_md,
)
from utils import reroute_cmd
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

type Then[**P, U] = Callable[P, U | Coroutine[Any, Any, U]]


async def _chained(prev: Awaitable, callbacks: Sequence[Then]) -> Any:
    out = await prev
    for callback in callbacks:
        log.debug('then: got %r, calling %r', out, callback)
        if out is None:
            out = callback()
        elif isinstance(out, (list, tuple)):
            out = callback(*out)
        elif isinstance(out, dict):
            out = callback(*out.values(), **out)
        else:
            out = callback(out)
        if isinstance(out, Promise):
            log.debug('then: awaiting task %r', out._task)
            out = await out._task
    if out is not None:
        log.debug('then: final result %r', out)
    return out


type PromiseResult = Value | list[Value] | tuple[Value, ...] | dict[str, Value] | None


class Promise[T: PromiseResult](Box):
    __slots__ = ('_task', '_factory', '_token')

    def __init__(
        self,
        coro: Awaitable[T],
        factory: Callable[[Awaitable[T]], 'Promise[T]'],
        token: object,
    ):
        self._task = create_task(self._run(coro))
        self._factory = factory
        self._token = token

    async def _run(self, coro: Awaitable[T]) -> T | None:
        log.debug('Promise: starting coroutine %r', coro)
        try:
            return await coro
        except Exception:
            log.exception('Promise: coroutine failed')
        finally:
            log.debug('Promise: resolved %r', coro)
            del self._token

    def then(self, *callbacks: Then) -> Promise:
        # `callback` is expected to be a `SubDoc` with a `scope`, so we can call
        # it safely while not rendering.
        if not callbacks:
            return self
        if not all(callable(f) for f in callbacks):
            raise TypeError(f'Promise.then: callback must be callable, got {callbacks}')
        return self._factory(_chained(self._task, callbacks))

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
            s = to_segment(v)
            # Exclude empty strings and sequences.
            if isinstance(s, str):
                # Join consecutive strings.
                if s:
                    parts.append(s)
            else:
                # Fragment is flattened when iterated, so returned seg cannot be
                # another Sequence.
                assert isinstance(s, BaseElement)
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

    if name in _funcs or name in _methods:  # noqa: F821
        raise ValueError(f'{func} is already registered')

    return name, is_method


def public[**P, R](func: Callable[P, R], name: str | None = None) -> Callable[P, R]:
    name, is_method = _inspect(func, name=name)
    if is_method:
        _methods[name] = None  # noqa: F821
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

        _methods[name] = None  # noqa: F821
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

        _methods[name] = wrapper2  # noqa: F821
        return wrapper2


class Callbacks(Protocol):
    async def _update_text(self, seg: Segment) -> Value | None: ...
    def _error(self, msg: str) -> Any: ...
    def _escalate(self) -> int | None: ...


class Bridge(Box):
    __slots__ = ('_ctx', '_trusted', '_cb_ref', '_promise_cap', '_temp_files')

    def __init__(
        self, ctx: MutableMapping[str, Value], trusted: int | None, cb: Callbacks
    ) -> None:
        super().__init__()
        self._ctx = ctx
        self._trusted = trusted
        self._cb_ref = weakref.ref(cb, self._finalize)
        self._promise_cap = 5 if trusted is None else 10
        self._temp_files = []

    def __repr__(self) -> str:
        return f'Bridge({self._trusted})'

    @property
    def _cb(self) -> Callbacks:
        if (r := self._cb_ref()) is None:
            raise RuntimeError('Bridge: context gone')
        return r

    def _finalize(self, ref):
        # Break the reference cycle, since the Context can hold references to
        # `Bridge` and `SubDoc` (which holds `Engine`).
        self._ctx.clear()
        cnt = 0
        for file in self._temp_files:
            try:
                os.remove(file)
            except OSError as e:
                log.error(
                    'Bridge: failed to remove temp file %r: %s: %s',
                    file,
                    type(e).__name__,
                    e,
                )
            else:
                cnt += 1
        if cnt:
            log.info('Bridge: removed %d temp files', cnt)
        else:
            log.debug('Bridge: removed %d temp files', cnt)

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

    def _promise[T: PromiseResult](self, coro: Awaitable[T]) -> Promise[T]:
        if self._promise_cap is not None:
            if self._promise_cap <= 0:
                raise RuntimeError('Promise capacity exceeded')
            self._promise_cap -= 1

        # Let each promise hold a reference to `self._cb` to keep it alive until
        # all promises are resolved.
        return Promise(coro, self._promise, self._cb)

    @trusted
    def communicate(self, cmd, input='') -> Promise[dict[str, Value]]:
        return self._promise(_communicate(to_str(cmd), to_str(input)))

    @trusted
    def mkstemp(self, *args, **kwargs) -> 'LocalPath':
        fd, path = tempfile.mkstemp(*args, **kwargs)
        self._temp_files.append(path)
        os.close(fd)
        return LocalPath(path)

    @trusted
    def evil(self, code):
        code = to_str(code).strip()
        if '\n' in code:
            res = []

            def print(*args):
                res.extend(repr(arg) for arg in args)

            exec(code, globals={'print': print}, locals=self._ctx)
            return '\n'.join(res)

        return eval(code, locals=self._ctx)

    @public
    def escalate(self) -> None:
        if (token := self._cb._escalate()) is not None:
            self._trusted = token

    @public
    def edit_message(self, text) -> Promise:
        log.debug('Bridge: edit_message: %r %r', text, self._cb)
        return self._promise(self._cb._update_text(to_segment(text)))

    @public
    def sleep(self, seconds: float) -> Promise[None]:
        if self._trusted is None and seconds > 60:
            raise ValueError('sleep: too long')
        return self._promise(asyncio.sleep(seconds))

    async def _reroute_cmd(
        self, cmd: str, coro: Awaitable[Sequence[tuple[str, str | None]]] | None
    ) -> Fragment[Raw | str] | Raw | str | None:
        if coro is None:
            self._cb._error(f'command not found: {cmd!r}')
            return None
        r = await coro
        log.info('Bridge: exec %r: %r', cmd, r)
        if len(r) == 1:
            text, parse_mode = r[0]
            # Avoid a repeated escaping.
            return Raw(text) if parse_mode == 'MarkdownV2' else text
        if r:
            return Fragment(
                [
                    Raw(text) if parse_mode == 'MarkdownV2' else text
                    for text, parse_mode in r
                ]
            )

    @public
    def exec(self, cmd) -> Promise:
        log.debug('Bridge: exec: %r', cmd)
        cmd = to_str(cmd)
        if not cmd.startswith('/'):
            raise ValueError(f'command must start with /, got {cmd!r}')
        coro = reroute_cmd(get_context().update, cmd)
        return self._promise(self._reroute_cmd(cmd, coro))

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


async def _communicate(cmd: str, input: str | None) -> dict[str, Value]:
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
    r = {
        'stdout': stdout,
        'stderr': stderr,
        'elapsed': elapsed,
    }
    if returncode is not None:
        r['returncode'] = returncode
    return r


class LocalPath(Box):
    __slots__ = ('path',)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __repr__(self):
        return f'LocalPath({self.path!r})'

    def __str__(self):
        return self.path


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


@public
def btoa(data) -> str:
    if isinstance(data, int):
        data = data.to_bytes((data.bit_length() + 7) // 8)
    elif not isinstance(data, bytes):
        data = to_str(data).encode()
    return base64.b64encode(data).decode()


@public
def atob(text) -> bytes:
    if not isinstance(text, bytes):
        text = to_str(text).encode('ascii')
    text += b'=' * (-len(text) % 4)
    return base64.b64decode(text)


@public
def cleanup(text) -> str:
    return cleanup_text(to_str(text))


@public
def cleanup2(text) -> str:
    return cleanup_text_md(to_str(text))


for name, func in _methods.items():
    if func is not None:
        setattr(Bridge, name, func)

del _methods
