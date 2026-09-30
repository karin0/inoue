import asyncio

from unittest.mock import MagicMock

import pytest

from inoue.render_ctx import RenderContext
from inoue.render_lib import Promise, Tasks

from .fakes import settle


def _ctx() -> MagicMock:
    return MagicMock(name='ctx', spec=RenderContext)


def test_empty_group_counts_zero():
    tasks = Tasks()
    assert not tasks
    assert tasks.count('k') == 0


async def test_finished_task_leaves_the_group_and_reports_done():
    tasks = Tasks()
    ctx = _ctx()

    async def coro():
        return 'done'

    task = asyncio.create_task(coro())
    tasks.create(task, ctx)
    assert tasks.count('k') == 1
    assert tasks

    await settle()
    assert tasks.count('k') == 0
    ctx._task_done.assert_called_once()


async def test_future_counts_only_its_chained_callbacks():
    tasks = Tasks()
    ctx = _ctx()
    fut1: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    fut2: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    p1 = tasks.create(fut1, ctx)
    p2 = tasks.create(fut2, ctx)
    assert tasks.count('k') == 0

    p1.then(lambda: None)
    p1.then(lambda: None)
    p2.then(lambda: None)
    assert tasks.count('k') == 3

    fut1.set_result(None)
    await settle()
    assert tasks.count('k') == 1

    fut2.set_result(None)
    await settle()
    assert tasks.count('k') == 0
    assert ctx._task_done.call_count == 2


async def test_cancel_cancels_every_task():
    tasks = Tasks()
    ctx = _ctx()
    pending = [asyncio.create_task(asyncio.sleep(10)) for _ in range(2)]
    for task in pending:
        tasks.create(task, ctx)
    assert tasks.count('k') == 2

    tasks.cancel()
    await settle()
    assert all(task.cancelled() for task in pending)
    assert ctx._task_done.call_count == 2


async def test_task_result_resolves_its_promise():
    seen: list[int] = []

    async def coro():
        return 42

    Tasks().create(asyncio.create_task(coro()), _ctx()).then(seen.append)
    await settle()
    assert seen == [42]


async def test_failed_task_resolves_its_promise_to_nothing():
    seen: list[tuple] = []

    async def boom():
        raise RuntimeError('boom')

    Tasks().create(asyncio.create_task(boom()), _ctx()).then(lambda *args: seen.append(args))
    await settle()
    assert seen == [()]


async def test_cancelled_task_skips_its_callbacks():
    seen: list[None] = []
    task = asyncio.create_task(asyncio.sleep(10))
    Tasks().create(task, _ctx()).then(seen.append)
    task.cancel()
    await settle()
    assert seen == []


def test_then_chains_in_order():
    seen: list[tuple[str, int]] = []
    p: Promise = Promise()

    def f(v):
        seen.append(('f', v))
        return v * 2

    def g(v):
        seen.append(('g', v))
        return v + 1

    p.then(f, g)
    p._resolve(3)
    assert seen == [('f', 3), ('g', 6)]


def test_then_on_a_resolved_promise_runs_immediately():
    seen: list[str] = []
    p: Promise = Promise()
    p._resolve('x')
    p.then(seen.append)
    assert seen == ['x']


def test_callback_returning_a_promise_waits_for_it():
    seen: list[str] = []
    inner: Promise = Promise()
    head: Promise = Promise(lambda _: inner)
    head.then(seen.append)
    head._resolve(1)
    assert seen == []

    inner._resolve('done')
    assert seen == ['done']


def test_promise_resolving_to_itself_is_a_loop():
    p: Promise = Promise(lambda _: p)
    with pytest.raises(ValueError, match='loop'):
        p._resolve(1)


def test_promises_resolving_to_each_other_are_circular():
    a: Promise = Promise(lambda _: b)
    b: Promise = Promise(lambda _: a)
    a._resolve(1)
    with pytest.raises(ValueError, match='circular'):
        b._resolve(1)


def test_cancel_propagates_to_chained_callbacks():
    seen: list[str] = []
    p: Promise = Promise()
    p.then(seen.append)
    p._cancel()
    assert seen == []


def test_then_rejects_non_callable():
    with pytest.raises(TypeError):
        Promise().then('x')  # pyright: ignore[reportArgumentType] -- the wrong type is the input
