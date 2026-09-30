from unittest.mock import AsyncMock, MagicMock

import pytest

from bot import message_responder
from bot.message_responder import MessageEditHandle
from inoue.ctx import use_context
from inoue.render import handle_render_callback

from .fakes import HOST, FakeResponder, settle


@pytest.fixture(autouse=True)
def _host():
    update = MagicMock(name='update', effective_user=None, effective_chat=None)
    update.effective_message = None
    with use_context(update, None, HOST):
        yield


class ChatResponder(FakeResponder):
    __slots__ = ()

    def as_edit_handle(self) -> MessageEditHandle:
        return MessageEditHandle(1, 2)


async def test_display_flag_applies(monkeypatch: pytest.MonkeyPatch):
    bot = MagicMock(name='bot')
    bot.edit_message_text = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(message_responder, 'bot', bot)
    await handle_render_callback(MagicMock(name='callback'), '+_plain`{_plain}', ChatResponder())
    await settle()
    call = bot.edit_message_text.await_args
    assert call.args[0].startswith('1')
    assert call.kwargs['parse_mode'] is None


@pytest.mark.parametrize('key', ['_trusted', '_chat_id', '_btn'])
def test_host_key_is_refused_as_flag(key: str):
    with pytest.raises(ValueError, match='host key'):
        handle_render_callback(MagicMock(name='callback'), f'+{key}`{{{key}}}', FakeResponder(''))
