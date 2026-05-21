import asyncio
import base64
import gc
import inspect
import os
import subprocess
import sys
import tempfile
import time

from collections import ChainMap
from collections.abc import Callable, Coroutine, MutableMapping
from datetime import datetime
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Concatenate, cast
from weakref import WeakValueDictionary, ref

from bot import Responder, escape, html_escape
from render_core import Box, Fragment, Value, to_str

from .log import log
from .motto import hitokoto
from .render_lib import INSTANCES, LocalPath, Promise, PromiseResult, Tasks, communicate, to_segment
from .segments import (
    BaseElement,
    BlockQuote,
    Bold,
    Code,
    Element,
    Italic,
    Link,
    Pre,
    Raw,
    Segment,
    Spoiler,
    Strikethrough,
    Style,
    Underline,
)
from .text import cleanup_text, cleanup_text_md
from .utils import reroute_cmd

if TYPE_CHECKING:
    from .render import RenderContext

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
    func: Callable[Concatenate[Bridge, P], R] | Callable[P, R], *, name: str | None = None
) -> Callable[Concatenate[Bridge, P], R]:
    name, is_method = _inspect(func, name=name)

    if is_method:
        func = cast(Callable[Concatenate['Bridge', P], R], func)

        @wraps(func)
        def wrapper(self: Bridge, *args: P.args, **kwargs: P.kwargs) -> R:
            if self._trusted is None:
                log.warning('Bridge: unauthorized access to %s', name)
                raise PermissionError('unauthorized')
            log.debug('Bridge: authorized %s for %s', self._trusted, name)
            return func(self, *args, **kwargs)

        _methods[name] = None  # noqa: F821
        return wrapper
    func = cast(Callable[P, R], func)

    @wraps(func)
    def wrapper2(self: Bridge, *args: P.args, **kwargs: P.kwargs) -> R:
        if self._trusted is None:
            log.warning('Bridge: unauthorized access to %s', name)
            raise PermissionError('unauthorized')
        log.debug('Bridge: authorized %s for %s', self._trusted, name)
        return func(*args, **kwargs)

    _methods[name] = wrapper2  # noqa: F821
    return wrapper2


TASK_GROUPS: WeakValueDictionary[str, Tasks] = WeakValueDictionary()


def count_tasks(key: str) -> int:
    if (tasks := TASK_GROUPS.get(key)) is not None:
        return tasks.count(key)
    return 0


class Bridge(Box):
    __slots__ = ('_ctx', '_trusted', '_cb_ref', '_promise_cap', '_temp_files', '_tasks')

    def __init__(
        self,
        ctx: MutableMapping[str, Value],
        trusted: int | None,
        task_policy: str | None,
        tasks_key: str | None,
        cb: RenderContext,
    ) -> None:
        super().__init__()
        self._ctx = ctx
        self._trusted = trusted
        self._cb_ref = ref(cb, self._finalize)
        self._promise_cap = 5 if trusted is None else 1000
        self._temp_files = []

        if tasks_key and (tasks := TASK_GROUPS.get(tasks_key)) is not None:
            log.debug('Bridge: remaining tasks: %s, %r', tasks_key, tasks)
            tasks.cancel()

        if task_policy:
            # Reason to reject.
            self._tasks = task_policy
        elif tasks_key:
            TASK_GROUPS[tasks_key] = tasks = Tasks()
            log.debug('Bridge: allowing tasks: %s', tasks_key)
            self._tasks = tasks
        else:
            raise TypeError('tasks_key is required when task_policy is None')

        INSTANCES.add(cb)

    def __repr__(self) -> str:
        return f'Bridge({self._trusted}, {self._cb_ref!r})'

    @property
    def _cb(self) -> RenderContext:
        if (r := self._cb_ref()) is None:
            raise RuntimeError('Bridge: context gone')
        return r

    def _finalize(self, _ref):
        # Break the reference cycle, since the Context can hold references to
        # `Bridge` and `SubDoc` (which holds `Engine`).
        self._ctx.clear()
        if isinstance(self._tasks, Tasks):
            self._tasks.cancel()
        cnt = 0
        for file in self._temp_files:
            try:
                os.remove(file)
            except OSError as e:
                log.error(
                    'Bridge: failed to remove temp file %r: %s: %s', file, type(e).__name__, e
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

    def _promise[T: PromiseResult](self, coro: Coroutine[Any, Any, T]) -> Promise[T]:
        if self._promise_cap is not None:
            if self._promise_cap <= 0:
                raise RuntimeError('Promise capacity exceeded')
            self._promise_cap -= 1

        if isinstance(tasks := self._tasks, Tasks):
            # Let each task hold a reference to the `RenderContext` to keep it
            # alive until all promises are resolved.
            return tasks.create(asyncio.create_task(coro), self._cb)
        raise RuntimeError(f'Promise: {tasks}')

    def _promise_fut[T: PromiseResult](self, fut: asyncio.Future[T]) -> Promise[T]:
        if self._promise_cap is not None:
            if self._promise_cap <= 0:
                raise RuntimeError('Promise capacity exceeded')
            self._promise_cap -= 1

        if isinstance(tasks := self._tasks, Tasks):
            return tasks.create(fut, self._cb)
        raise RuntimeError(f'Promise: {tasks}')

    @trusted
    def communicate(self, cmd, input='') -> Promise[dict[str, Value]]:
        return self._promise(communicate(to_str(cmd), to_str(input)))

    @trusted
    def mkstemp(self, *args, **kwargs) -> LocalPath:
        if isinstance(self._tasks, str):
            raise RuntimeError(self._tasks)
        fd, path = tempfile.mkstemp(*args, **kwargs)
        self._temp_files.append(path)
        os.close(fd)
        return LocalPath(path)

    @trusted
    def evil(self, code):
        code = to_str(code).strip()

        # Bring back the names overridden by `RenderContext`.
        ctx: MutableMapping[str, Any] = self._ctx
        local = ChainMap({'os': os, 'sys': sys}, ctx)
        if '\n' in code:
            res = []

            def print(*args):
                res.extend(map(repr, args))

            exec(code, globals={'print': print}, locals=local)  # noqa: S102
            return '\n'.join(res)

        return eval(code, locals=local)  # noqa: S307

    @trusted
    def debug(self, *vals) -> Promise:
        text = '\n'.join(repr(val) for val in vals)
        return self._promise(self._cb._reply(text))

    @public
    def escalate(self) -> None:
        if (token := self._cb._escalate()) is not None:
            self._trusted = token

    @public
    def edit_message(self, text) -> Promise:
        log.debug('Bridge: edit_message: %r %r', text, self._cb)
        return self._promise_fut(self._cb._edit_message(text))

    @public
    def sleep(self, seconds: float) -> Promise[None]:
        if self._trusted is None and seconds > 60:
            raise ValueError('sleep: too long')
        return self._promise(asyncio.sleep(seconds))

    async def _reroute_cmd(self, rs: Responder, cmd: str) -> Fragment[Raw | str] | Raw | str | None:
        r = await reroute_cmd(rs, cmd)
        log.info('Bridge: exec %r: %r', cmd, r)
        if r is None:
            self._cb._error(f'exec failed: {cmd!r}')
            return None
        if len(r) == 1:
            text, parse_mode = r[0]
            # Avoid a repeated escaping.
            return Raw(text) if parse_mode == 'MarkdownV2' else text
        if r:
            return Fragment(
                [Raw(text) if parse_mode == 'MarkdownV2' else text for text, parse_mode in r]
            )

    @public
    def exec(self, cmd) -> Promise:
        log.debug('Bridge: exec: %r', cmd)
        cmd = to_str(cmd)
        if not cmd.startswith('/'):
            raise ValueError(f'command must start with /, got {cmd!r}')
        if (rs := self._cb._responder) is None:
            raise RuntimeError('No responder')
        return self._promise(self._reroute_cmd(rs, cmd))

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
    result = subprocess.check_output(  # noqa: S602
        cmd,
        shell=True,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=0.1,
        env={'LANG': 'C', 'LC_ALL': 'C'},
    )
    return result.strip()


@trusted
def top(do_gc: bool = False) -> Fragment[str]:
    if do_gc:
        gc.collect()
    n = len(INSTANCES)
    m = len(TASK_GROUPS)
    info = f'{n} instances, {m} task groups'
    log.debug('top: %s', info)
    return Fragment(
        [
            info + '\n',
            *sorted(f'[{id(o)} {sys.getrefcount(o)}] {o!r}\n' for o in INSTANCES),
            *(f'[{id(o)} {sys.getrefcount(o)} {k}]: {o!r}\n' for k, o in TASK_GROUPS.items()),
        ]
    )


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
    def inner(self) -> Segment:  # pyright: ignore[reportIncompatibleVariableOverride]
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
