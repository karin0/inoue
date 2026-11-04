import asyncio

from typing import TYPE_CHECKING

import pytest

from inoue import shell

if TYPE_CHECKING:
    from pathlib import Path

# CI has no systemd user instance, so the command runs without a scope and the
# test records which scopes inoue stops.


@pytest.fixture
def stopped(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    stopped: list[str] = []

    async def stop(unit: str):
        stopped.append(unit)

    monkeypatch.setattr(shell, 'scope_argv', lambda _unit, _secs, command: ['bash', '-c', command])
    monkeypatch.setattr(shell, 'stop_scope', stop)
    return stopped


async def test_the_output_merges_both_streams_and_ends_with_the_status(stopped: list[str]):
    out = await shell.run('echo out; echo err >&2; exit 3', 5, '/')
    assert out == 'out\nerr\n\n[exit status 3]'
    assert len(stopped) == 1


async def test_a_long_output_keeps_its_head_and_tail(stopped: list[str]):
    n = shell.KEEP_BYTES
    out = await shell.run(f'printf a%.0s $(seq {n}); printf b%.0s $(seq {n + 7}); printf c', 5, '/')
    head, tail = out.split('\n[8 bytes omitted]\n')
    assert head == 'a' * n
    assert tail == 'b' * (n - 1) + 'c\n[exit status 0]'


async def test_a_hung_command_times_out_and_is_stopped(
    monkeypatch: pytest.MonkeyPatch, stopped: list[str]
):
    monkeypatch.setattr(shell, 'GRACE_SECS', 0.1)
    with pytest.raises(TimeoutError):
        await shell.run('exec sleep 30', 0, '/')
    assert len(stopped) == 1


async def test_a_cancelled_turn_stops_the_command(stopped: list[str]):
    task = asyncio.create_task(shell.run('exec sleep 30', 5, '/'))
    await asyncio.sleep(0.2)
    task.cancel()
    async with asyncio.timeout(2):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(stopped) == 1


def test_the_scope_keeps_dollar_signs_for_bash():
    argv = shell.scope_argv('u', 5, 'echo $HOME')
    assert '--expand-environment=no' in argv
    assert argv[-3:] == ['bash', '-c', 'echo $HOME']


async def test_the_tool_refuses_a_turn_without_the_shell(stopped: list[str], tmp_path: Path):
    allowed = False
    tool = shell.shell_tool(str(tmp_path), lambda: allowed)
    refused = await tool.handler({'command': 'touch x'})
    assert refused['is_error']
    assert not (tmp_path / 'x').exists()

    allowed = True
    assert not (await tool.handler({'command': 'touch x'}))['is_error']
    assert (tmp_path / 'x').exists()
    assert len(stopped) == 1


@pytest.mark.parametrize('secs', [0, shell.MAX_SECS + 1])
async def test_the_tool_rejects_a_timeout_out_of_range(stopped: list[str], secs: int):
    tool = shell.shell_tool('/', lambda: True)
    assert (await tool.handler({'command': 'true', 'timeout': secs}))['is_error']
    assert stopped == []


def test_the_command_environment_keeps_paths_and_drops_secrets():
    assert 'TELEGRAM_BOT_TOKEN' not in shell.ENV
    assert shell.ENV['TZ'] == 'Asia/Tokyo'
