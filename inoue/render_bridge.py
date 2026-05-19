import asyncio
import base64
import gc
import inspect
import os
import subprocess
import sys
import tempfile
import time

from collections.abc import Callable, Coroutine, MutableMapping
from datetime import datetime
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Concatenate, cast
from weakref import WeakSet, WeakValueDictionary, ref

from bot import Responder, escape, html_escape
from render_core import Box, Fragment, Value, to_str

from .log import log
from .motto import hitokoto
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

type PromiseResult = Value | list[Value] | tuple[Value, ...] | dict[str, Value] | None

type Callback[T: PromiseResult] = Callable[..., T | Promise[T] | None]


def _call_then[T: PromiseResult](callback: Callback[T], arg) -> T | Promise[T] | None:
    # `callback` is expected to be a `SubDoc` with a `scope`, so we can call
    # it safely while not rendering.
    log.debug('then: got %r, calling %r', arg, callback)
    if arg is None:
        r = callback()
    elif isinstance(arg, (list, tuple)):
        r = callback(*arg)
    elif isinstance(arg, dict):
        r = callback(*arg.values(), **arg)
    else:
        r = callback(arg)
    return r


class Promise[T: PromiseResult](Box):
    __slots__ = ('_inner', '_func', '_next', '__weakref__')

    def __init__(self, func: Callback[T] | None = None):
        self._inner: list[Promise] | tuple[T | Promise[T] | None] = []
        self._func = func
        self._next: WeakSet[Promise] = WeakSet()
        INSTANCES.add(self)

    def _resolve(self, prev: PromiseResult):
        log.debug('Promise._resolve: %r\n prev: %r', self, prev)

        if isinstance(prev, Promise):
            prev._chain(self)
            return

        if self._func is not None:
            result = _call_then(self._func, prev)
            log.debug('Promise._resolve: returned: %r\n -> %r', self, result)
        else:
            # When `_func` is None, the input type must be the same as the output
            # type `T`.
            result = cast(T | None, prev)

        del self._func
        inner = self._inner
        if not isinstance(inner, list):
            self._cancel()
            raise RuntimeError('Promise._resolve: already resolved: %r %r', self, inner)

        # We track all our "resolved" successors, i.e. `v` such that
        # `v._inner == (self,)` in `self._next` to collapse the waiting chain.
        self._inner = r = (result,)
        for v in self._next:
            v._inner = r

        if isinstance(result, Promise):
            # Technically "resolved", but actually still one in-degree produced
            # by `_func`.
            # We relink our successors to that predecessor to avoid leaks from a
            # growing waiting chain.
            if result is self:
                # A self loop. Do not clear `_next`, leave them getting cancelled.
                self._cancel()
                log.error('Promise._resolve: loop chaining: %r', inner)
                raise ValueError('loop chaining')

            for v in self._next:
                if v is not result:
                    result._next.add(v)
                else:
                    # A 2-cycle. Further detection is skipped.
                    self._cancel()
                    v._cancel()
                    log.error('Promise._resolve: circular chaining: %r', inner)
                    raise ValueError('circular chaining')

            result._next.add(self)
            self._next.clear()
            for cb in inner:
                result._chain(cb)
        else:
            self._next.clear()
            for cb in inner:
                cb._resolve(result)

    def _cancel(self):
        log.debug('Promise._cancel: %r', self)
        inner = self._inner
        self._inner = r = (None,)
        for v in self._next:
            v._inner = r
        self._next.clear()
        if isinstance(inner, list):
            for cb in inner:
                cb._cancel()

    def _invoke(self, prev: asyncio.Task[T | Promise[T]]) -> None:
        log.debug('Promise: invoke %r\n task: %s', self, _format_task(prev))
        if prev.cancelled():
            self._cancel()
            return
        try:
            r = prev.result()
        except Exception as e:
            log.error(
                'Promise: exception: %r, %r: %s: %s', self, prev, type(e).__name__, e, exc_info=e
            )
            self._resolve(None)
        else:
            self._resolve(r)

    def _chain(self, fut: Promise):
        inner = self._inner
        if isinstance(inner, list):
            inner.append(fut)
        else:
            fut._resolve(inner[0])

    def then(self, *funcs: tuple[Callback, ...]) -> Promise:
        promise = self
        if not all(callable(f) for f in funcs):
            raise TypeError(f'Promise.then: callback must be callable, got {funcs}')
        for f in funcs:
            f: Any
            # p = p._then(f)
            new = Promise(f)
            promise._chain(new)
            promise = new
        return promise

    def __repr__(self) -> str:
        return _repr(self, set())


def _repr_list(obj: list, vis: set[int]) -> str:
    return f'[{", ".join(_repr(v, vis) for v in obj)}]'


def _repr(obj, vis: set[int]) -> str:
    if not isinstance(obj, Promise):
        return repr(obj)
    x = id(obj)
    if x in vis:
        return '<Promise(...)>'
    vis.add(x)
    try:
        inner = obj._inner
        n = len(obj._next)
        if isinstance(inner, list):
            return (
                f'<Promise({len(inner)}/{n}): {_repr_list(inner, vis)} '
                f'func={getattr(obj, "_func", "/")}>'
            )
        return f'<Promise(resolved/{n}): {_repr(inner[0], vis)}>'
    finally:
        vis.remove(x)


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


def _format_task(task: asyncio.Task) -> str:
    if task.cancelled():
        state = 'cancelled'
    elif task.done():
        try:
            r = task.result()
        except Exception as e:
            state = f'exception: {type(e).__name__}: {e}'
        else:
            state = f'done: {r!r}' if (r := task.result()) is not None else 'done'
    else:
        state = 'pending'

    name = coro.__qualname__ if (coro := task.get_coro()) is not None else '<unknown>'

    return f'<{task.get_name()} ({name}): {state}>'


class Tasks:
    __slots__ = ('_tasks', '__weakref__')

    def __init__(self):
        self._tasks: set[asyncio.Task] = set()

    def __repr__(self) -> str:
        return f'Tasks[{", ".join(_format_task(t) for t in self._tasks)}]'

    def __bool__(self) -> bool:
        return bool(self._tasks)

    def create[T: PromiseResult](
        self, coro: Coroutine[Any, Any, T], ctx: RenderContext
    ) -> Promise[T]:
        promise = Promise()

        def callback(fut: asyncio.Task[T], _=ctx):
            self._tasks.remove(fut)
            try:
                promise._invoke(fut)
            except ValueError as e:
                # Circular chaining detected.
                promise._cancel()
                log.warning('Tasks.callback: %s: %s', type(e).__name__, e)
                ctx._error(f'Promise: {e}')
            ctx.flush_errors()

        task = asyncio.create_task(coro)
        task.add_done_callback(callback)
        self._tasks.add(task)
        return promise

    def cancel(self):
        log.debug('Tasks.cancel: %r', self)
        if tasks := self._tasks:
            for task in tasks:
                task.cancel()
            asyncio.gather(*tasks, return_exceptions=True).add_done_callback(self._cancel_done)

    def _cancel_done(self, fut: asyncio.Future):
        log.debug('Tasks._cancel_done: %r: %r', self, fut.result())
        for task in self._tasks:
            if not task.done():
                log.warning('Tasks._cancel_done: task not done: %r', _format_task(task))


INSTANCES = WeakSet()
TASK_GROUPS: WeakValueDictionary[str, Tasks] = WeakValueDictionary()


def count_tasks(key: str) -> int:
    if (tasks := TASK_GROUPS.get(key)) is not None:
        return len(tasks._tasks)
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
            return tasks.create(coro, self._cb)
        raise RuntimeError(f'Promise: {tasks}')

    @trusted
    def communicate(self, cmd, input='') -> Promise[dict[str, Value]]:
        return self._promise(_communicate(to_str(cmd), to_str(input)))

    @trusted
    def mkstemp(self, *args, **kwargs) -> 'LocalPath':  # noqa: UP037
        if isinstance(self._tasks, str):
            raise RuntimeError(self._tasks)
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

            exec(code, globals={'print': print}, locals=self._ctx)  # noqa: S102
            return '\n'.join(res)

        return eval(code, locals=self._ctx)  # noqa: S307

    @public
    def escalate(self) -> None:
        if (token := self._cb._escalate()) is not None:
            self._trusted = token

    @public
    def edit_message(self, text) -> Promise:
        log.debug('Bridge: edit_message: %r %r', text, self._cb)
        return self._promise(self._cb._edit_message(to_segment(text)))

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
    except TimeoutError:
        proc.terminate()
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        except TimeoutError:
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
    r = {'stdout': stdout, 'stderr': stderr, 'elapsed': elapsed}
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
