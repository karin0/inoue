import asyncio
import gc
import os
import weakref

from itertools import chain
from typing import TYPE_CHECKING

import pytest

from inoue import render_bridge
from inoue.render_bridge import TASK_GROUPS, count_tasks
from inoue.render_ctx import BUTTON_KEY, MessageSpec

from .fakes import GUEST, HOST, FakeResponder, bridge_of, make_ctx, make_data, settle

if TYPE_CHECKING:
    from collections.abc import Callable


async def test_guest_promise_capacity_is_five():
    ctx, _ = make_ctx(sender=GUEST, responder=FakeResponder())
    bridge = bridge_of(ctx)
    for _ in range(5):
        bridge.edit_message('x')
    with pytest.raises(RuntimeError, match='capacity'):
        bridge.edit_message('x')


async def test_trusted_promise_capacity_exceeds_five():
    ctx, _ = make_ctx(sender=HOST, responder=FakeResponder())
    bridge = bridge_of(ctx)
    for _ in range(6):
        bridge.edit_message('x')


def test_mkstemp_needs_an_editable_context():
    ctx, _ = make_ctx(sender=HOST)
    with pytest.raises(RuntimeError, match='uneditable context'):
        bridge_of(ctx).mkstemp()


def test_collected_context_removes_temp_files_and_clears_data():
    ctx, _ = make_ctx(sender=HOST, responder=FakeResponder())
    data = ctx.data
    bridge = bridge_of(ctx)
    paths = [bridge.mkstemp().path, bridge.mkstemp().path]
    assert all(os.path.exists(p) for p in paths)

    del ctx, bridge
    gc.collect()
    assert not any(os.path.exists(p) for p in paths)
    assert not data


async def test_context_dropped_with_a_pending_edit_is_finalized_after_sending_it():
    ctx, seen = make_ctx(sender=HOST, responder=FakeResponder())
    bridge = bridge_of(ctx)
    path = bridge.mkstemp().path
    bridge.edit_message('x')
    ref = weakref.ref(ctx)

    del ctx, bridge
    await settle()
    assert len(seen) == 1
    assert ref() is None
    assert not os.path.exists(path)


async def test_flushed_edit_keeps_its_context_alive_for_its_callbacks():
    ctx, _ = make_ctx(responder=FakeResponder())
    ref = weakref.ref(ctx)
    alive: list[bool] = []
    bridge_of(ctx).edit_message('x').then(lambda _: alive.append(ref() is not None))
    response = ctx.to_response('seed')

    del ctx
    await response
    await settle()
    assert alive == [True]


async def test_pending_task_keeps_its_context_alive():
    rs = FakeResponder()
    ctx, _ = make_ctx(responder=rs)
    bridge_of(ctx).sleep(60)
    ref = weakref.ref(ctx)
    del ctx
    gc.collect()
    assert ref() is not None

    TASK_GROUPS[rs.get_message_key()].cancel()
    await settle()
    gc.collect()
    assert ref() is None


async def test_rerender_rejects_edits_from_callbacks_already_scheduled():
    rs = FakeResponder()
    ctx, seen = make_ctx(responder=rs)
    bridge = bridge_of(ctx)
    rejected: list[str] = []

    def edit():
        try:
            bridge.edit_message('stale')
        except RuntimeError as e:
            rejected.append(str(e))

    async def finish():
        pass

    bridge._promise(finish()).then(edit)
    # The task finishes and schedules its done callback, which a cancel can no longer stop.
    await asyncio.sleep(0)
    make_ctx(responder=rs)

    await settle()
    assert not any('stale' in spec[0] for spec in seen)
    assert rejected


def _has_cancel_button(spec: MessageSpec) -> bool:
    markup = spec[2]
    return markup is not None and any(b.text == '🛑1' for b in chain(*markup.inline_keyboard))


async def test_chained_edit_loop_sends_one_update_per_round():
    ctx, seen = make_ctx(responder=FakeResponder())
    bridge = bridge_of(ctx)
    rounds = iter(('one', 'two', 'three'))

    def write(_=None):
        if (text := next(rounds, None)) is not None:
            bridge.edit_message(text).then(write)

    write()
    await ctx.to_response('seed')
    await settle()
    assert len(seen) == 3
    texts = ('seedone', 'two', 'three')
    assert all(text in spec[0] for text, spec in zip(texts, seen, strict=True))
    assert all(_has_cancel_button(spec) for spec in seen)


async def test_cancel_button_stops_an_edit_loop():
    rs = FakeResponder()
    ctx, seen = make_ctx(sender=HOST, responder=rs)
    bridge = bridge_of(ctx)
    rejected: list[str] = []

    def guard(call: Callable[[], object]):
        try:
            call()
        except RuntimeError as e:
            rejected.append(str(e))

    def tick(_=None):
        guard(lambda: bridge.sleep(0).then(edit))

    def edit():
        guard(lambda: bridge.edit_message('tick').then(tick))

    tick()
    await ctx.to_response('seed')
    for _ in range(20):
        await asyncio.sleep(0)
    sent = len(seen)
    assert sent >= 3
    assert not rejected

    make_ctx(make_data({BUTTON_KEY: '_cancel'}), responder=rs)
    await settle()
    assert len(seen) == sent
    assert set(rejected) <= {'Promise: context cancelled'}


async def test_count_tasks_drains_as_tasks_finish():
    rs = FakeResponder()
    ctx, _ = make_ctx(responder=rs)
    bridge = bridge_of(ctx)
    bridge.sleep(0)
    bridge.sleep(0)
    assert count_tasks(rs.get_message_key()) == 2
    await settle()
    assert count_tasks(rs.get_message_key()) == 0


def test_untrusted_context_cannot_call_trusted_functions():
    ctx, _ = make_ctx(sender=GUEST)
    bridge = bridge_of(ctx)
    with pytest.raises(PermissionError, match='unauthorized'):
        bridge.evil('1+1')
    with pytest.raises(PermissionError):
        bridge.uname()


def test_evil_sees_the_real_os_module():
    ctx, _ = make_ctx(sender=HOST)
    assert bridge_of(ctx).evil("os.path.basename('/a/b')") == 'b'


def test_evil_statements_leave_context_untouched():
    ctx, _ = make_ctx(make_data({'x': '42'}), sender=HOST)
    bridge = bridge_of(ctx)
    assert bridge.evil('y = x\nprint(y)') == "'42'"
    assert 'y' not in ctx.data


def test_language_cannot_reach_underscored_names():
    ctx, _ = make_ctx(sender=HOST)
    bridge = bridge_of(ctx)
    assert bridge._get_func('_finalize') is None
    assert bridge._get_func('escape_') is None
    assert bridge._get_func('escape') is not None


def test_module_functions_are_attributes():
    ctx, _ = make_ctx()
    bridge = bridge_of(ctx)
    assert bridge.cleanup('a') == 'a'
    with pytest.raises(AttributeError):
        _ = bridge.does_not_exist


def test_duplicate_registration_is_rejected():
    def cleanup(text):
        return text

    with pytest.raises(ValueError, match='already registered'):
        render_bridge.public(cleanup)
