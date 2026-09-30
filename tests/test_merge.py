import asyncio

from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram import InputMediaPhoto, InputMediaVideo, Message
from telegram.constants import ReactionEmoji

from inoue import merge
from inoue.ctx import use_context
from inoue.handlers import handle_msg
from inoue.merge import Key, Session, handle_merge

from .fakes import HOST, FakeResponder


@pytest.fixture(autouse=True)
def _host():
    with use_context(MagicMock(name='update'), None, HOST):
        yield


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> dict[Key, Session]:
    sessions: dict[Key, Session] = {}
    monkeypatch.setattr(merge, 'merge_sessions', sessions)
    return sessions


@pytest.fixture
def send_media_group(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    send = AsyncMock(return_value=[])
    monkeypatch.setattr(merge.bot, 'send_media_group', send)
    return send


def key_of(msg: Message) -> Key:
    key = merge.session_key(msg)
    assert key is not None
    return key


def _media(message_id: int, **kinds) -> MagicMock:
    kinds = {'photo': None, 'video': None, 'audio': None, 'document': None, **kinds}
    return MagicMock(
        message_id=message_id,
        caption='c',
        caption_entities=None,
        forward_origin=MagicMock(name='origin'),
        delete=AsyncMock(),
        reply_text=AsyncMock(),
        **kinds,
    )


async def test_first_merge_starts_a_session_that_expires(
    sessions: dict[Key, Session], monkeypatch: pytest.MonkeyPatch
):
    loop = asyncio.get_running_loop()
    real_call_later = loop.call_later
    scheduled = []

    def call_later(delay, callback, *args):
        handle = real_call_later(delay, callback, *args)
        scheduled.append((delay, callback, args, handle))
        return handle

    monkeypatch.setattr(loop, 'call_later', call_later)
    rs = FakeResponder('/merge')
    msg = rs.get_message()
    await handle_merge(rs, msg, '')

    key = key_of(msg)
    assert 'Merge Start' in (rs.replies[0][0] or '')
    assert sessions[key].messages == []

    [(delay, callback, args, handle)] = scheduled
    handle.cancel()
    assert delay == 300
    callback(*args)
    assert key not in sessions


async def test_merge_cancel_drops_the_session(sessions: dict[Key, Session]):
    rs = FakeResponder('/merge cancel')
    msg = rs.get_message()
    timer = MagicMock()
    sessions[key_of(msg)] = Session([MagicMock()], timer)

    await handle_merge(rs, msg, 'cancel')
    assert 'cancelled' in (rs.replies[0][0] or '')
    assert not sessions
    timer.cancel.assert_called_once()


async def test_media_message_joins_the_session(sessions: dict[Key, Session]):
    rs = FakeResponder()
    msg = rs.get_message()
    msg.photo = (MagicMock(file_id='photo_1'),)
    msg.video = msg.audio = msg.document = None
    msg.set_reaction = AsyncMock(return_value=True)
    session = sessions[key_of(msg)] = Session([], MagicMock())

    await handle_msg(rs)
    assert session.messages == [msg]
    msg.set_reaction.assert_called_once_with(ReactionEmoji.RED_HEART)
    assert rs.replies == []


async def test_second_merge_sends_the_group_in_message_order(
    sessions: dict[Key, Session], send_media_group: AsyncMock
):
    rs = FakeResponder('/merge')
    msg = rs.get_message()
    photo = _media(10, photo=(MagicMock(file_id='file1'),))
    video = _media(11, video=MagicMock(file_id='file2'))
    timer = MagicMock()
    sessions[key_of(msg)] = Session([video, photo], timer)

    await handle_merge(rs, msg, 'Album Title')
    timer.cancel.assert_called_once()
    send_media_group.assert_called_once()
    kwargs = send_media_group.call_args.kwargs
    assert kwargs['caption'] == 'Album Title'
    first, second = kwargs['media']
    assert isinstance(first, InputMediaPhoto)
    assert first.media == 'file1'
    assert first.caption is None
    assert isinstance(second, InputMediaVideo)
    assert second.media == 'file2'
    assert second.caption is None

    assert not sessions
    assert rs.replies == []
    photo.delete.assert_called_once()
    video.delete.assert_called_once()


async def test_single_item_is_not_sent(sessions: dict[Key, Session], send_media_group: AsyncMock):
    rs = FakeResponder('/merge')
    msg = rs.get_message()
    photo = _media(10, photo=(MagicMock(file_id='file1'),))
    sessions[key_of(msg)] = Session([photo], MagicMock())

    await handle_merge(rs, msg, '')
    send_media_group.assert_not_called()
    assert not sessions
    assert rs.replies == []
    photo.reply_text.assert_called_once()
