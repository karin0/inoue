import asyncio
import time

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakSet

from render_core import Box, Fragment, Value, to_str

from .log import log
from .segments import BaseElement

if TYPE_CHECKING:
    from .render import RenderContext


def join_str[T](iter: Iterable[T]) -> Iterable[T]:
    # Join consecutive strings.
    buf = []
    for v in iter:
        if isinstance(v, str):
            buf.append(v)
        else:
            if buf:
                if r := ''.join(buf):
                    yield r  # type: ignore[yield-type]
                buf.clear()
            yield v
    if buf and (r := ''.join(buf)):
        yield r  # type: ignore[yield-type]


# This must have more than one element, though the type system cannot enforce it.
type PluralSequence[T] = Sequence[T]
type FlattenSegment = str | BaseElement | PluralSequence[str | BaseElement]


def to_segment(val: Value | None) -> FlattenSegment:
    if val is None:
        return ''

    if isinstance(val, BaseElement):
        return val

    if isinstance(val, Fragment):
        # Fragment is flattened when iterated, so returned seg cannot be
        # another Sequence.
        out: list[BaseElement | str] = [to_segment(v) for v in join_str(val)]  # type: ignore[assignment]
        if len(out) == 1:
            return out[0]
        return out or ''

    return to_str(val)


def merge_segments(segs: Iterable[FlattenSegment]) -> FlattenSegment:
    out: list[str | BaseElement] = []
    for seg in join_str(segs):
        if isinstance(seg, (str, BaseElement)):
            out.append(seg)
        else:
            out.extend(seg)
    if len(out) == 1:
        return out[0]
    return out or ''


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


INSTANCES = WeakSet()


class Promise[T: PromiseResult](Box):
    __slots__ = ('_inner', '_func', '_next', '__weakref__')

    def __init__(self, func: Callback[T] | None = None, then_cb: Callable[[], Any] | None = None):
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

            if result in self._next:
                # A 2-cycle. Further detection is skipped.
                self._cancel()
                result._cancel()
                log.error('Promise._resolve: circular chaining: %r', inner)
                raise ValueError('circular chaining')

            result._next.update(self._next)
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

    def _invoke(self, prev: asyncio.Future[T | Promise[T]]) -> None:
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


def _format_task(task: asyncio.Future) -> str:
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

    if isinstance(task, asyncio.Task):
        name = coro.__qualname__ if (coro := task.get_coro()) is not None else '<unknown>'

        return f'<{task.get_name()} ({name}): {state}>'
    return f'<Future: {state}>'


class Tasks:
    __slots__ = ('_tasks', '_promises', '__weakref__')

    def __init__(self):
        self._tasks: set[asyncio.Future] = set()
        self._promises: WeakSet[Promise] = WeakSet()

    def __repr__(self) -> str:
        return (
            f'Tasks[tasks: {", ".join(_format_task(t) for t in self._tasks)}, '
            f'promises: {", ".join(repr(p) for p in self._promises)}]'
        )

    def __bool__(self) -> bool:
        return bool(self._tasks)

    def create[T: PromiseResult](self, task: asyncio.Future[T], ctx: RenderContext) -> Promise[T]:
        promise = Promise()
        if not isinstance(task, asyncio.Task):
            # For non-task futures from edit_message, we count their chained promises instead, or
            # a cancel button will always show up for every edited message.
            self._promises.add(promise)

        def callback(fut: asyncio.Future[T], _=ctx):
            # `set.remove` may raise here, since the `Future` produced by `edit_message` can be
            # chained to multiple promises.
            self._tasks.discard(fut)
            self._promises.discard(promise)
            try:
                promise._invoke(fut)
            except ValueError as e:
                # Circular chaining detected.
                promise._cancel()
                log.warning('Tasks.callback: %s: %s', type(e).__name__, e)
                ctx._error(f'Promise: {e}')
            ctx._task_done()

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

    def count(self, key: str) -> int:
        n = len([t for t in self._tasks if isinstance(t, asyncio.Task) and not t.done()])
        m = sum(len(p._inner) for p in self._promises if isinstance(p._inner, list))
        log.debug(
            'Tasks.count: %s: %r: %d/%d tasks, %d/%d promises',
            key,
            self,
            n,
            len(self._tasks),
            m,
            len(self._promises),
        )
        return n + m


async def communicate(cmd: str, input: str | None) -> dict[str, Value]:
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
