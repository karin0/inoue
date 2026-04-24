import os
import sys
import time
import asyncio
import subprocess

from dataclasses import dataclass
from typing import Awaitable, Callable, cast
from collections.abc import MutableMapping

from render_core import Box, Value

from util import log, escape, html_escape, pre_block_raw, cleanup_text
from motto import hitokoto


async def _run[T](coro: Awaitable[T]) -> T | None:
    try:
        return await coro
    except Exception:
        log.exception('Promise: coroutine failed')
        return None


async def _chained[T, U](
    prev: Awaitable[T], callback: Callable[[T], U | Awaitable[U]]
) -> U:
    result = await prev
    log.debug('Promise: calling callback %r with result %r', callback, result)
    out = callback(result)
    if asyncio.iscoroutine(out):
        log.debug('Promise: awaiting callback result %r', out)
        out = await out
    return cast(U, out)


class Promise[T: Value | None](Box):
    def __init__(self, coro: Awaitable[T]):
        super().__init__()
        self._task = asyncio.create_task(_run(coro))

    def then[U: Value | None](
        self, callback: Callable[[T | None], U | Awaitable[U]]
    ) -> Promise[U]:
        # `callback` is expected to be a `SubDoc` with a `scope`, so we can call
        # it safely while not rendering.
        if not callable(callback):
            raise TypeError(f'Promise.then: callback must be callable, got {callback}')
        return Promise(_chained(self._task, callback))

    def __repr__(self) -> str:
        return f'<Promise task={self._task!r}>'


_PUBLIC: set[str] = set()
_TRUSTED: set[str] = set()


def public(method):
    _PUBLIC.add(method.__name__)
    return method


def trusted(method):
    _TRUSTED.add(method.__name__)
    return method


class Bridge(Box):
    def __init__(
        self,
        ctx: MutableMapping[str, Value],
        update_text: Callable[[str], Awaitable[Value | None]],
        trusted: int | None = None,
    ) -> None:
        super().__init__()
        self._ctx = ctx
        self._update_text = update_text
        self._trusted = trusted

        funcs: dict[str, Callable[..., Value | None]] = {
            name: getattr(self, name) for name in _PUBLIC
        }
        for name in _TRUSTED:
            funcs[name] = self._auth(name, getattr(self, name))

        self.funcs = funcs

    def __repr__(self) -> str:
        return f'<Bridge signature={self._trusted}>'

    def _auth[**P, R](self, name: str, func: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs):
            if self._trusted is None:
                log.warning('Bridge: unauthorized access to %s', name)
                raise PermissionError('unauthorized')
            log.debug('Bridge: authorized %s for %s', self._trusted, name)
            return func(*args, **kwargs)

        return wrapper

    def __getattr__(self, name: str) -> Callable[..., Value | None]:
        funcs = self.__dict__.get('funcs')
        if funcs is not None and name in funcs:
            return funcs[name]
        raise AttributeError(f'bad bridge call: {name}')

    @trusted
    def system(self, cmd: str) -> str:
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
    def eval(self, expr: str):
        return eval(expr)

    @trusted
    def uname(self) -> str:
        r = os.uname()
        return f'{r.sysname} {r.nodename} {r.release} {r.version} {r.machine}'

    @trusted
    def version(self) -> str:
        return sys.version

    @trusted
    def write_file(self, path, text) -> None:
        if not isinstance(path, str):
            raise TypeError(f'write_file: path must be a str, got {path!r}')

        with open(path, 'w', encoding='utf-8') as fp:
            fp.write(str(text))

    @trusted
    def edit_message(self, text) -> Promise:
        return Promise(self._update_text(str(text)))

    @trusted
    def communicate(self, cmd: str, input: str | None = None) -> Promise[ProcessResult]:
        return Promise(_communicate(cmd, input))

    @trusted
    def sleep(self, seconds: float) -> Promise[None]:
        return Promise(asyncio.sleep(seconds))

    @public
    def dbg(self) -> str:
        return '\n'.join(f'{k}={v!r}' for k, v in self._ctx.items())

    @public
    def escape(self, text) -> str:
        return escape(str(text))

    @public
    def html_escape(self, text) -> str:
        return html_escape(str(text))

    @public
    def pre(self, text) -> str:
        return pre_block_raw(str(text))

    @public
    def mono(self, text) -> str:
        return f'`{escape(str(text))}`'

    @public
    def cleanup(self, text) -> str:
        return cleanup_text(str(text))

    @public
    def hitokoto(self):
        return hitokoto()


@dataclass
class ProcessResult(Box):
    stdout: str
    stderr: str
    returncode: int | None
    elapsed: float


async def _communicate(cmd: str, input: str | None = None) -> ProcessResult:
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdin=asyncio.subprocess.PIPE if input else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input.encode('utf-8') if input else None),
            timeout=10,
        )
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

    elapsed = time.monotonic() - t0
    stdout = stdout.decode(errors='replace').strip()
    stderr = stderr.decode(errors='replace').strip()
    return ProcessResult(stdout, stderr, returncode, elapsed)
