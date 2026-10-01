import asyncio

import pytest

from inoue.env import USER_ID
from inoue.render_bridge import TASK_GROUPS, count_tasks
from inoue.render_ctx import BUTTON_KEY, MessageSpec

from .fakes import GUEST, HOST, FakeResponder, bridge_of, make_ctx, make_data, settle


def test_context_without_update_callback_is_uneditable():
    ctx, _ = make_ctx()
    with pytest.raises(RuntimeError, match='uneditable'):
        bridge_of(ctx).edit_message('hi')


def test_editable_context_registers_its_task_group():
    rs = FakeResponder()
    ctx, _ = make_ctx(responder=rs)
    assert TASK_GROUPS[rs.get_message_key()] is bridge_of(ctx)._tasks


async def test_cancel_button_rejects_new_promises():
    ctx, _ = make_ctx(make_data({BUTTON_KEY: '_cancel'}), responder=FakeResponder())
    with pytest.raises(RuntimeError, match='context cancelled'):
        bridge_of(ctx).edit_message('hi')


async def test_rejected_edit_is_not_sent():
    ctx, seen = make_ctx(make_data({BUTTON_KEY: '_cancel'}), responder=FakeResponder())
    with pytest.raises(RuntimeError, match='context cancelled'):
        bridge_of(ctx).edit_message('rejected')
    await ctx.to_response('seed')
    assert 'rejected' not in seen[0][0]


def test_host_sender_is_trusted():
    ctx, _ = make_ctx(sender=HOST)
    assert ctx.data['_trusted'] == USER_ID
    assert bridge_of(ctx).evil('1 + 1') == 2


def test_guest_sender_is_untrusted():
    ctx, _ = make_ctx(sender=GUEST)
    assert '_trusted' not in ctx.data
    with pytest.raises(PermissionError):
        bridge_of(ctx).evil('1 + 1')


def test_escalate_trusts_a_saved_doc():
    ctx, _ = make_ctx(sender=GUEST, doc_id=42)
    bridge = bridge_of(ctx)
    bridge.escalate()
    assert ctx.data['_trusted'] == 42
    assert bridge.evil('1 + 1') == 2


def test_escalate_without_doc_is_rejected():
    ctx, _ = make_ctx(sender=GUEST)
    bridge = bridge_of(ctx)
    bridge.escalate()
    assert '_trusted' not in ctx.data
    with pytest.raises(PermissionError):
        bridge.evil('1 + 1')


async def test_edits_in_one_callback_share_a_task_and_one_update():
    rs = FakeResponder()
    ctx, seen = make_ctx(responder=rs)
    task = ctx._edit_message('A')
    assert ctx._edit_message('B') is task
    assert ctx._edit_message('C') is task

    assert await task == rs.get_message_key()
    assert len(seen) == 1
    text = seen[0][0]
    assert 'A' in text
    assert 'B' in text
    assert 'C' in text


async def test_edit_to_none_empties_the_message():
    ctx, seen = make_ctx(responder=FakeResponder())
    await ctx._edit_message(None)
    assert 'empty' in seen[0][0]


async def test_cancelled_edit_skips_the_update():
    ctx, seen = make_ctx(responder=FakeResponder())
    ctx._edit_message('z').cancel()
    await settle()
    assert seen == []


async def test_first_response_goes_through_the_update_callback():
    rs = FakeResponder()
    ctx, seen = make_ctx(responder=rs)
    await ctx.to_response('hello')
    assert len(seen) == 1
    assert 'hello' in seen[0][0]
    assert ctx._edit_handle is rs.handle


async def test_edit_is_sent_after_the_running_callback():
    ctx, seen = make_ctx(responder=FakeResponder())
    ctx._edit_message('hello')
    assert seen == []
    await settle()
    assert len(seen) == 1
    assert 'hello' in seen[0][0]


async def test_task_done_replies_new_errors_once():
    rs = FakeResponder()
    ctx, seen = make_ctx(responder=rs)
    await ctx.to_response('seed')

    ctx.engine.errors.append('something went wrong')
    ctx._task_done()
    await settle()
    assert [r[0] for r in rs.replies] == ['something went wrong']

    ctx._task_done()
    await settle()
    assert len(rs.replies) == 1
    assert len(seen) == 1


async def test_task_done_without_edits_or_errors_sends_nothing():
    rs = FakeResponder()
    ctx, seen = make_ctx(responder=rs)
    ctx._task_done()
    await settle()
    assert seen == []
    assert rs.replies == []


async def test_rerender_of_the_same_message_cancels_old_tasks():
    rs1 = FakeResponder()
    old, _ = make_ctx(responder=rs1, sender=HOST)
    bridge_of(old).sleep(60)
    assert count_tasks(rs1.get_message_key()) == 1

    rs2 = FakeResponder(message_id=rs1.get_message().message_id)
    new, _ = make_ctx(responder=rs2)
    await settle()
    assert count_tasks(rs1.get_message_key()) == 0
    assert TASK_GROUPS[rs2.get_message_key()] is bridge_of(new)._tasks


async def test_rendered_edit_message_merges_into_the_response():
    ctx, seen = make_ctx(responder=FakeResponder())
    seg = ctx.render_text("{ edit_message('hi') }tail")
    await ctx.to_response(seg)
    assert len(seen) == 1
    text = seen[0][0]
    assert 'hi' in text
    assert 'tail' in text


async def test_render_merges_edits_even_if_awaited_late():
    ctx, seen = make_ctx(responder=FakeResponder())
    response = ctx.render("{ edit_message('hi') }tail")
    await asyncio.sleep(0)
    await response
    assert len(seen) == 1
    assert 'hi' in seen[0][0]
    assert 'tail' in seen[0][0]


def _buttons(spec: MessageSpec) -> list[str]:
    markup = spec[2]
    return [b.text for row in markup.inline_keyboard for b in row] if markup else []


async def test_cancel_button_shows_only_for_work_chained_after_an_edit():
    ctx, seen = make_ctx(responder=FakeResponder())
    bridge = bridge_of(ctx)
    bridge.edit_message('plain')
    await settle()
    bridge.edit_message('chained').then(lambda _: None)
    await settle()
    assert not any(b.startswith('🛑') for b in _buttons(seen[0]))
    assert '🛑1' in _buttons(seen[1])


async def test_edit_message_promise_resolves_to_the_message_key():
    rs = FakeResponder()
    ctx, _ = make_ctx(responder=rs)
    keys: list[str] = []
    bridge_of(ctx).edit_message('payload').then(keys.append)
    await ctx.to_response('seed')
    await settle()
    assert keys == [rs.get_message_key()]


def test_repr_names_source_and_message():
    ctx, _ = make_ctx(make_data({'_source': 'me', '_chat_id': 1, '_msg_id': 2}))
    assert 'me (1, 2)' in repr(ctx)
