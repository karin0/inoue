from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram.constants import ChatType

from inoue import voice
from inoue.ctx import use_context
from inoue.handlers import SKIP_MARK, handle_msg

from .fakes import HOST, FakeResponder


@pytest.fixture(autouse=True)
def _host():
    with use_context(MagicMock(name='update'), None, HOST):
        yield


@pytest.fixture
def convert(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    convert = AsyncMock()
    monkeypatch.setattr(voice, '_handle_voice', convert)
    return convert


def reply_to_voice(text: str) -> FakeResponder:
    rs = FakeResponder(text)
    msg = rs.get_message()
    msg.chat.type = ChatType.PRIVATE
    msg.via_bot = None
    msg.forward_origin = None
    msg.document = msg.audio = msg.video = msg.voice = None
    msg.reply_to_message = MagicMock(document=None, audio=None, video=None)
    return rs


async def test_reply_to_voice_converts_it(convert: AsyncMock):
    await handle_msg(reply_to_voice('nc'))
    convert.assert_called_once()


@pytest.mark.parametrize('text', [f'{SKIP_MARK} nice', f'nice {SKIP_MARK}'])
async def test_skip_mark_leaves_the_message_alone(convert: AsyncMock, text: str):
    rs = reply_to_voice(text)
    await handle_msg(rs)
    convert.assert_not_called()
    assert rs.replies == []
