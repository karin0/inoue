from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram.constants import ChatType

from inoue import handlers, voice, ytdlp
from inoue.ai import Mode
from inoue.ctx import use_context
from inoue.env import USER_ID
from inoue.handlers import SKIP_MARK, handle_msg
from inoue.voice import VoiceArgs

from .fakes import HOST, FakeResponder


@pytest.fixture(autouse=True)
def _host():
    update = MagicMock(name='update')
    update.effective_chat.type = ChatType.PRIVATE
    update.effective_chat.id = USER_ID
    with use_context(update, None, HOST):
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
    msg.sticker = msg.animation = msg.video_note = None
    msg.photo = ()
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


@pytest.mark.parametrize(
    ('text', 'url'),
    [
        ('BV1GJ411x7h7', 'https://www.bilibili.com/video/BV1GJ411x7h7'),
        (' dQw4w9WgXcQ ', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'),
        ('https://youtu.be/dQw4w9WgXcQ', 'https://youtu.be/dQw4w9WgXcQ'),
        ('av117465963037923 q', 'https://www.bilibili.com/video/av117465963037923'),
    ],
)
async def test_video_alone_or_with_q_downloads_it(
    monkeypatch: pytest.MonkeyPatch, text: str, url: str
):
    download = AsyncMock()
    monkeypatch.setattr(ytdlp, 'run_ytdlp', download)
    await handle_msg(_private(text))
    download.assert_called_once_with(url, audio_only=False)


@pytest.mark.parametrize('text', ['BV1GJ411x7h7 nice', 'see https://youtu.be/dQw4w9WgXcQ'])
async def test_video_with_other_words_is_searched(monkeypatch: pytest.MonkeyPatch, text: str):
    download, search = AsyncMock(), AsyncMock()
    monkeypatch.setattr(ytdlp, 'run_ytdlp', download)
    monkeypatch.setattr(handlers, 'handle_rg', search)
    await handle_msg(_private(text))
    download.assert_not_called()
    search.assert_called_once()


@pytest.mark.parametrize(
    ('text', 'url'),
    [
        ('BV1GJ411x7h7 nc', 'https://www.bilibili.com/video/BV1GJ411x7h7'),
        ('https://youtu.be/dQw4w9WgXcQ 10- q', 'https://youtu.be/dQw4w9WgXcQ'),
    ],
)
async def test_video_with_parameters_converts_it(convert: AsyncMock, text: str, url: str):
    await handle_msg(_private(text))
    convert.assert_called_once()
    assert convert.call_args.args[1:3] == (None, url)


@pytest.fixture
def claude(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    claude = AsyncMock()
    monkeypatch.setattr(handlers, 'handle_ai', claude)
    monkeypatch.setitem(handlers.MODES, Mode.CHAT, MagicMock())
    return claude


def host_guest(text: str) -> FakeResponder:
    rs = _private(text)
    msg = rs.get_message()
    msg.guest_query_id = 'guest'
    msg.reply_to_message = MagicMock(
        name='photo',
        photo=(MagicMock(),),
        sticker=None,
        document=None,
        audio=None,
        video=None,
        voice=None,
    )
    return rs


@pytest.mark.parametrize('text', ['what is this', 'see https://youtu.be/dQw4w9WgXcQ'])
async def test_host_text_in_a_guest_chat_goes_to_claude(claude: AsyncMock, text: str):
    rs = host_guest(text)
    await handle_msg(rs, direct=False)
    claude.assert_awaited_once_with(rs, rs.get_message(), text)


async def test_host_lone_url_in_a_guest_chat_downloads_it(
    monkeypatch: pytest.MonkeyPatch, claude: AsyncMock
):
    download = AsyncMock()
    monkeypatch.setattr(ytdlp, 'run_ytdlp', download)
    rs = host_guest('https://youtu.be/dQw4w9WgXcQ')
    rs.get_message().reply_to_message = None
    await handle_msg(rs, direct=False)
    claude.assert_not_called()
    download.assert_called_once_with('https://youtu.be/dQw4w9WgXcQ', audio_only=False)


async def test_host_video_with_q_in_a_guest_chat_downloads_it(
    monkeypatch: pytest.MonkeyPatch, claude: AsyncMock
):
    download = AsyncMock()
    monkeypatch.setattr(ytdlp, 'run_ytdlp', download)
    rs = host_guest('av117465963037923 q')
    rs.get_message().reply_to_message = None
    await handle_msg(rs, direct=False)
    claude.assert_not_called()
    download.assert_called_once_with(
        'https://www.bilibili.com/video/av117465963037923', audio_only=False
    )


async def test_host_text_in_a_guest_chat_converts_media_without_claude(
    monkeypatch: pytest.MonkeyPatch,
):
    claude, sticker = AsyncMock(), AsyncMock(return_value=True)
    monkeypatch.setattr(handlers, 'handle_ai', claude)
    monkeypatch.setattr(handlers, 'handle_sticker', sticker)
    rs = host_guest('what is this')
    await handle_msg(rs, direct=False)
    claude.assert_not_called()
    sticker.assert_awaited_once()


@pytest.mark.parametrize(('text', 'claude_called'), [('nc 10-', False), ('what is said', True)])
async def test_host_reply_to_voice_in_a_guest_chat_converts_only_parameters(
    convert: AsyncMock, claude: AsyncMock, text: str, claude_called: bool
):
    rs = reply_to_voice(text)
    rs.get_message().guest_query_id = 'guest'
    await handle_msg(rs, direct=False)
    assert claude.await_count == claude_called
    assert convert.call_count == (not claude_called)


async def test_host_rembg_in_a_guest_chat_converts_the_sticker(
    monkeypatch: pytest.MonkeyPatch, claude: AsyncMock
):
    sticker = AsyncMock(return_value=True)
    monkeypatch.setattr(handlers, 'handle_sticker', sticker)
    await handle_msg(host_guest('rembg'), direct=False)
    claude.assert_not_called()
    sticker.assert_awaited_once()


def reply_to_answer(text: str) -> FakeResponder:
    rs = _private(text)
    msg = rs.get_message()
    msg.guest_query_id = 'guest'
    answer = msg.reply_to_message = MagicMock(name='answer', text='an answer')
    answer.from_user.id = handlers.bot.bot.id
    return rs


@pytest.mark.parametrize('text', ['', 'rembg', 'nc 10-', 'https://youtu.be/dQw4w9WgXcQ'])
async def test_host_reply_to_an_answer_with_media_goes_to_claude(claude: AsyncMock, text: str):
    rs = reply_to_answer(text)
    rs.get_message().voice = MagicMock(name='voice')
    await handle_msg(rs, direct=False)
    claude.assert_awaited_once_with(rs, rs.get_message(), text)


async def test_host_bare_reply_to_an_answer_skips_claude(convert: AsyncMock, claude: AsyncMock):
    await handle_msg(reply_to_answer(''), direct=False)
    claude.assert_not_called()


async def test_host_video_with_parameters_in_a_guest_chat_converts_it(
    convert: AsyncMock, claude: AsyncMock
):
    rs = host_guest('https://youtu.be/dQw4w9WgXcQ nc')
    rs.get_message().reply_to_message = None
    await handle_msg(rs, direct=False)
    claude.assert_not_called()
    assert args_of(convert) == VoiceArgs(speed=1.5)
