import asyncio
import os
import uuid

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from claude_agent_sdk import create_sdk_mcp_server, tool

from .env import ME_LOWER
from .log import log

if TYPE_CHECKING:
    from collections.abc import Callable

    from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool

SERVER = 'host'
TOOL = f'mcp__{SERVER}__shell'

DEFAULT_SECS = 30
MAX_SECS = 600
# inoue waits this much longer than the scope may run, and as long for
# systemctl.
GRACE_SECS = 10
KEEP_BYTES = 15_000
# The commands see the user's paths, locale and session bus, and none of the
# secrets that inoue's own configuration puts in its environment.
_ENV_NAMES = frozenset(('PATH', 'HOME', 'USER', 'LANG', 'TZ', 'DBUS_SESSION_BUS_ADDRESS'))
ENV = MappingProxyType(
    {k: v for k, v in os.environ.items() if k in _ENV_NAMES or k.startswith(('XDG_', 'LC_'))}
)
# `systemctl stop` exits with this when the scope already ended.
NOT_LOADED = 5


class ShellArgs(TypedDict):
    command: str
    timeout: NotRequired[int]


class Output:
    '''The first and the last `KEEP_BYTES` of a stream, and how many bytes
    between them were dropped.'''

    __slots__ = ('head', 'tail', 'dropped')

    def __init__(self):
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    def add(self, data: bytes):
        if (room := KEEP_BYTES - len(self.head)) > 0:
            self.head += data[:room]
            data = data[room:]
        self.tail += data
        if (over := len(self.tail) - KEEP_BYTES) > 0:
            del self.tail[:over]
            self.dropped += over

    def text(self) -> str:
        cut = f'\n[{self.dropped} bytes omitted]\n' if self.dropped else ''
        return self.head.decode(errors='replace') + cut + self.tail.decode(errors='replace')


def scope_argv(unit: str, secs: int, command: str) -> list[str]:
    # The scope's cgroup holds every process the command starts, so systemd
    # kills them all after `secs` even when inoue has crashed.
    return [
        'systemd-run',
        '--user',
        '--scope',
        '--collect',
        '--quiet',
        '--expand-environment=no',
        f'--property=RuntimeMaxSec={secs}',
        f'--unit={unit}',
        '--',
        'bash',
        '-c',
        command,
    ]


async def stop_scope(unit: str):
    proc = await asyncio.create_subprocess_exec(
        'systemctl',
        '--user',
        'stop',
        f'{unit}.scope',
        env=ENV,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(GRACE_SECS):
            _, err = await proc.communicate()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode not in (0, NOT_LOADED):
        log.warning('shell: stop %s: %s', unit, err.decode(errors='replace').strip())


async def _read(stream: asyncio.StreamReader, out: Output):
    while chunk := await stream.read(1 << 16):
        out.add(chunk)


async def run(command: str, secs: int, cwd: str) -> str:
    '''Runs `command` until its shell exits, and then stops every process it
    left behind.'''
    unit = f'{ME_LOWER}-shell-{uuid.uuid4()}'
    proc = await asyncio.create_subprocess_exec(
        *scope_argv(unit, secs, command),
        cwd=cwd,
        env=ENV,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    out = Output()
    reader = asyncio.create_task(_read(proc.stdout, out))
    try:
        async with asyncio.timeout(secs + GRACE_SECS):
            code = await proc.wait()
    finally:
        try:
            await stop_scope(unit)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            try:
                async with asyncio.timeout(GRACE_SECS):
                    await reader
            except TimeoutError:
                log.warning('shell: %s: output still open', unit)
    status = f'exit status {code}' if code >= 0 else f'killed by signal {-code}'
    return f'{out.text()}\n[{status}]'


def _result(text: str, *, error: bool = False) -> dict[str, Any]:
    return {'content': [{'type': 'text', 'text': text}], 'is_error': error}


def shell_tool(cwd: str, allowed: Callable[[], bool]) -> SdkMcpTool[ShellArgs]:
    @tool(
        'shell',
        f'Run a bash command on the host in {cwd}, with stdin closed, and return its '
        'stdout and stderr. The turn waits for the command, which is killed after '
        f'`timeout` seconds, {DEFAULT_SECS} by default and at most {MAX_SECS}, and every '
        'process it leaves behind is killed when it exits.',
        ShellArgs,
    )
    async def shell(args: ShellArgs) -> dict[str, Any]:
        if not allowed():
            return _result('The owner did not allow shell commands in this turn.', error=True)
        secs = args.get('timeout', DEFAULT_SECS)
        if not 0 < secs <= MAX_SECS:
            return _result(f'timeout must be between 1 and {MAX_SECS}.', error=True)
        return _result(await run(args['command'], secs, cwd))

    return shell


def server(cwd: str, allowed: Callable[[], bool]) -> McpSdkServerConfig:
    return create_sdk_mcp_server(SERVER, tools=[shell_tool(cwd, allowed)])
