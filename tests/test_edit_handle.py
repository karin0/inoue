from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram import Message
from telegram.error import BadRequest

from bot import message_responder
from bot.message_responder import InlineMessageEditHandle, MessageEditHandle


@pytest.fixture
def bot(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    bot = MagicMock(name='bot')
    monkeypatch.setattr(message_responder, 'bot', bot)
    return bot


async def test_chat_handle_returns_the_edited_message(bot: MagicMock):
    edited = MagicMock(spec=Message)
    bot.edit_message_text = AsyncMock(return_value=edited)
    assert await MessageEditHandle(1, 2).edit_text('hi') is edited
    bot.edit_message_text.assert_awaited_once()
    assert bot.edit_message_text.await_args.kwargs['chat_id'] == 1
    assert bot.edit_message_text.await_args.kwargs['message_id'] == 2
    assert bot.edit_message_text.await_args.kwargs['inline_message_id'] is None


async def test_caption_handle_edits_the_caption(bot: MagicMock):
    bot.edit_message_caption = AsyncMock(return_value=MagicMock(spec=Message))
    await MessageEditHandle(1, 2, as_caption=True).edit_text('hi')
    assert bot.edit_message_caption.await_args.kwargs['caption'] == 'hi'


async def test_inline_handle_edits_by_inline_message_id(bot: MagicMock):
    bot.edit_message_text = AsyncMock(return_value=True)
    assert await InlineMessageEditHandle('abc').edit_text('hi') is True
    kwargs = bot.edit_message_text.await_args.kwargs
    assert (kwargs['chat_id'], kwargs['message_id'], kwargs['inline_message_id']) == (
        None,
        None,
        'abc',
    )


async def test_not_modified_yields_false_only_when_allowed(bot: MagicMock):
    bot.edit_message_text = AsyncMock(side_effect=BadRequest('Message is not modified'))
    handle = MessageEditHandle(1, 2)
    assert await handle.edit('hi', allow_not_modified=True) is False
    with pytest.raises(BadRequest):
        await handle.edit('hi')


async def test_edit_without_content_is_refused(bot: MagicMock):
    with pytest.raises(TypeError):
        await InlineMessageEditHandle('abc').edit()
