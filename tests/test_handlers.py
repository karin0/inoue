from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram.constants import ChatType

from inoue import voice
from inoue.ctx import use_context
from inoue.handlers import SKIP_MARK, handle_msg
from inoue.voice import VoiceArgs

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


def _private(text: str) -> FakeResponder:
    rs = FakeResponder(text)
    msg = rs.get_message()
    msg.chat.type = ChatType.PRIVATE
    msg.via_bot = None
    msg.forward_origin = None
    msg.document = msg.audio = msg.video = msg.voice = None
    return rs


def reply_to_voice(text: str) -> FakeResponder:
    rs = _private(text)
    rs.get_message().reply_to_message = MagicMock(document=None, audio=None, video=None)
    return rs


def voice_with_caption(caption: str) -> FakeResponder:
    rs = _private(caption)
    rs.get_message().voice = MagicMock(name='voice')
    return rs


def args_of(convert: AsyncMock) -> VoiceArgs:
    convert.assert_called_once()
    return convert.call_args.args[3]


async def test_reply_to_voice_converts_it_with_the_parameters(convert: AsyncMock):
    await handle_msg(reply_to_voice('nc 10-'))
    assert args_of(convert) == VoiceArgs(speed=1.5, start_time=10)


async def test_reply_to_voice_that_chats_leaves_it_alone(convert: AsyncMock):
    rs = reply_to_voice('sounds nice')
    await handle_msg(rs)
    convert.assert_not_called()
    assert rs.replies == []


async def test_caption_that_chats_converts_with_defaults(convert: AsyncMock):
    await handle_msg(voice_with_caption('sounds nice'))
    assert args_of(convert) == VoiceArgs()


async def test_command_with_bad_parameters_replies_usage(convert: AsyncMock):
    rs = reply_to_voice('/voice sounds nice')
    await handle_msg(rs)
    convert.assert_not_called()
    assert 'Send or reply' in (rs.replies[0][0] or '')


@pytest.mark.parametrize(
    'rs', [reply_to_voice(f'{SKIP_MARK} nc'), voice_with_caption(f'sounds nice {SKIP_MARK}')]
)
async def test_skip_mark_leaves_the_message_alone(convert: AsyncMock, rs: FakeResponder):
    await handle_msg(rs)
    convert.assert_not_called()
    assert rs.replies == []
