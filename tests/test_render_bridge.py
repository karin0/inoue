import gc
import os
import weakref

import pytest

from inoue import render_bridge
from inoue.render_bridge import TASK_GROUPS, count_tasks

from .fakes import GUEST, HOST, FakeResponder, bridge_of, make_ctx, make_data, settle


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
