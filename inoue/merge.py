import asyncio

from dataclasses import dataclass
from typing import TYPE_CHECKING

from telegram import (
    InputMediaAudio,
    InputMediaDocument,
    InputMediaPhoto,
    InputMediaVideo,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.constants import MediaGroupLimit, ReactionEmoji

from bot import MessageArg, Responder, bot, command

from .log import log

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence

type GroupMedia = InputMediaPhoto | InputMediaVideo | InputMediaAudio | InputMediaDocument
type Key = tuple[int, int, int | None]

MAX_QUEUE_MEDIA = 100


@dataclass(slots=True)
class Session:
    messages: list[Message]
    timer: asyncio.TimerHandle


merge_sessions: dict[Key, Session] = {}


def session_key(msg: Message) -> Key | None:
    if (sender := msg.sender_chat or msg.from_user) is not None:
        return msg.chat_id, sender.id, msg.message_thread_id


def clean_session(key: Key) -> None:
    merge_sessions.pop(key, None)
    log.info('Merge session %s timed out and was cleaned up.', key)


def check_merge(msg: Message) -> Coroutine | None:
    if (
        (msg.photo or msg.video or msg.audio or msg.document)
        and (key := session_key(msg)) is not None
        and (session := merge_sessions.get(key)) is not None
        and len(session.messages) < MAX_QUEUE_MEDIA
    ):
        session.messages.append(msg)
        return msg.set_reaction(ReactionEmoji.RED_HEART)


async def send_media_items[T: GroupMedia](
    key: Key, caption: str | None, items: Sequence[tuple[Message, T]]
) -> bool:
    if len(items) < MediaGroupLimit.MIN_MEDIA_LENGTH:
        await items[0][0].reply_text(
            '⚠️ A media group requires at least 2 files. 1 file was not merged.',
            do_quote=True,
            allow_sending_without_reply=True,
        )
        return False

    chat_id, _, thread_id = key
    i = 0
    while i < len(items):
        if len(items) - i < MediaGroupLimit.MIN_MEDIA_LENGTH:
            await items[i][0].reply_text(
                '⚠️ 1 file was omitted because a media group requires at least 2 files.',
                do_quote=True,
                allow_sending_without_reply=True,
            )
            return False
        chunk = tuple(
            items[j][1] for j in range(i, min(i + MediaGroupLimit.MAX_MEDIA_LENGTH, len(items)))
        )
        await bot.send_media_group(
            caption=caption, chat_id=chat_id, media=chunk, message_thread_id=thread_id
        )
        i += len(chunk)
    return True


@command(public=True)
async def handle_merge(rs: Responder, msg: Message, arg: MessageArg):
    if (key := session_key(msg)) is None:
        raise ValueError('handle_merge: no sender')

    if (session := merge_sessions.pop(key, None)) is None:
        timer = asyncio.get_running_loop().call_later(300, clean_session, key)
        merge_sessions[key] = Session([], timer)
        return await rs.reply(
            '📥 **Media Merge Start\\!**\n\n'
            'Send messages with photos, videos, audio, or documents here\\.\n'
            'When finished, send `/merge` again to combine them as a media group\\.\n'
            r'Send `/merge cancel` to cancel\.',
            parse_mode='MarkdownV2',
            reply_markup=ReplyKeyboardMarkup(
                ((('/merge', '/merge cancel'),)), one_time_keyboard=True, resize_keyboard=True
            ),
        )

    session.timer.cancel()
    messages = session.messages

    if arg.lower() == 'cancel' or not messages:
        return await rs.reply('Media Merge cancelled.', reply_markup=ReplyKeyboardRemove())

    # Sort messages by message_id to preserve the original sending/forwarding order
    messages.sort(key=lambda m: m.message_id)

    visuals: list[tuple[Message, InputMediaPhoto | InputMediaVideo]] = []
    audios: list[tuple[Message, InputMediaAudio]] = []
    docs: list[tuple[Message, InputMediaDocument]] = []

    def push[T: GroupMedia](lst: list[tuple[Message, T]], file_id: str, ty: type[T]):
        lst.append((m, ty(media=file_id, caption=caption, caption_entities=entities)))

    for m in messages:
        if arg:
            caption = entities = None
        else:
            caption = m.caption
            entities = m.caption_entities

        if m.photo:
            push(visuals, m.photo[-1].file_id, InputMediaPhoto)
        elif m.video is not None:
            push(visuals, m.video.file_id, InputMediaVideo)
        elif m.audio is not None:
            push(audios, m.audio.file_id, InputMediaAudio)
        elif m.document is not None:
            push(docs, m.document.file_id, InputMediaDocument)
        else:
            raise ValueError(f'No media in merge session: {m}')

    done = True
    caption = None if arg in ('', 'q', 's') else arg
    try:
        if visuals:
            done &= await send_media_items(key, caption, visuals)
        if audios:
            done &= await send_media_items(key, caption, audios)
        if docs:
            done &= await send_media_items(key, caption, docs)
    except Exception as e:
        log.exception('send_media_items failed')
        return await rs.reply(
            f'❌ Failed to send media group: {e}', reply_markup=ReplyKeyboardRemove()
        )

    if done:
        for m in reversed(messages):
            if m.forward_origin is not None or m.caption is None:
                await m.delete()
