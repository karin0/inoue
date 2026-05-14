from contextlib import contextmanager
from contextvars import ContextVar
from io import BytesIO

from telegram import (
    Message,
    InlineKeyboardMarkup,
    InputMedia,
    InputMediaDocument,
    InputMediaPhoto,
)
from typing import Awaitable, BinaryIO, ContextManager, Protocol, NamedTuple

from telegram.error import BadRequest

from .log import log
from .app import bot
from .env import CHAN_ID, GROUP_ID, USER_ID

from db import db


def encode_chat_id(m: Message, default: str = 'u') -> str:
    chat_id = m.chat_id
    if chat_id == USER_ID:
        return default
    if chat_id == CHAN_ID:
        return 'c'
    if chat_id == GROUP_ID:
        return 'g'
    return f'G{chat_id}'


reroute_capture: ContextVar[tuple[int, int, list[tuple[str, str | None]]] | None] = (
    ContextVar('reroute_capture', default=None)
)


class MediaPayload(Protocol):
    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> Message: ...

    def as_input(
        self, caption: str | None, parse_mode: str | None
    ) -> ContextManager[InputMedia] | None: ...


def open_payload(data: bytes | str) -> BinaryIO:
    if isinstance(data, bytes):
        return BytesIO(data)
    return open(data, 'rb')


class PhotoPayload(NamedTuple):
    photo: bytes | str

    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> Message:
        with open_payload(self.photo) as fp:
            return await msg.reply_photo(
                fp,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                do_quote=True,
                allow_sending_without_reply=True,
            )

    @contextmanager
    def as_input(self, caption: str | None, parse_mode: str | None):
        with open_payload(self.photo) as fp:
            yield InputMediaPhoto(fp, caption=caption, parse_mode=parse_mode)


class DocumentPayload(NamedTuple):
    document: bytes | str

    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> Message:
        with open_payload(self.document) as fp:
            return await msg.reply_document(
                fp,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                do_quote=True,
                allow_sending_without_reply=True,
            )

    @contextmanager
    def as_input(self, caption: str | None, parse_mode: str | None):
        with open_payload(self.document) as fp:
            yield InputMediaDocument(fp, caption=caption, parse_mode=parse_mode)


class Responder:
    def __init__(self, msg: Message):
        self.msg = msg

    async def reply(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        cached: bool = False,
        media: MediaPayload | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Message | bool:
        m = self.msg
        if not isinstance(m, Message):
            # XXX: Skip for `InlineMessageProxy`, since we won't receive updates for
            # edited guest messages anyway.
            # After we migrate to `Responder` entirely, maybe we can turn `InlineMessageProxy`
            # into another `Responder` and remove this check.
            key = 'X'
            cached = False
        else:
            key = f'{encode_chat_id(m)}-{m.message_id}'

        def _reply_text(text: str) -> Awaitable[Message]:
            return m.reply_text(
                text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                disable_web_page_preview=disable_web_page_preview,
                do_quote=True,
                allow_sending_without_reply=True,
            )

        # Do not call `_do_reply` with `save` set when `db[key]` presents.
        async def _do_reply(save: bool = True) -> Message:
            if as_caption := media is not None:
                try:
                    resp = await media.reply(m, text, parse_mode, reply_markup)
                except BadRequest as e:
                    if 'too long' in str(e):
                        # TODO
                        log.info(
                            '_do_reply: too long for caption, fallback to text: %s', e
                        )
                        resp = await media.reply(m, None, None, None)
                        if text:
                            resp = await _reply_text(text)
                        as_caption = False
                    else:
                        raise
            elif text:
                resp = await _reply_text(text)
            else:
                raise TypeError('Either text or media must be provided')

            if save:
                val = str(resp.message_id)
                if as_caption:
                    val = '@' + val
                db[key] = val
                log.debug('_do_reply: %s -> %s', key, val)
            return resp

        if (
            (reroute := reroute_capture.get()) is not None
            and reroute[0] == m.chat_id
            and reroute[1] == m.message_id
        ):
            log.info('reroute_capture: %s', key)
            if text:
                reroute[2].append((text, parse_mode))

            # Do not try to edit the reply, or we will mess up the response of the
            # capturing context (`/render`).
            return await _do_reply(False)

        if not cached:
            return await _do_reply(False)

        if not (val := db.get(key)):
            return await _do_reply()

        if media is None:
            input_media = None
        else:
            input_media = media.as_input(text, parse_mode)
            if input_media is None:
                db.discard(key)
                return await _do_reply()

        if val[0] == '@':
            as_caption = True
            resp_msg_id = int(val[1:])
        else:
            as_caption = False
            resp_msg_id = int(val)

        log.debug('Editing cached response: %s -> %s', key, val)

        try:
            try:
                if input_media is not None:
                    with input_media as im:
                        return await bot.edit_message_media(
                            im,
                            m.chat.id,
                            resp_msg_id,
                            reply_markup=reply_markup,
                        )
                if not text:
                    raise TypeError('Either text or media must be provided')
                if as_caption:
                    return await bot.edit_message_caption(
                        m.chat.id,
                        resp_msg_id,
                        caption=text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                return await bot.edit_message_text(
                    text,
                    m.chat.id,
                    resp_msg_id,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_web_page_preview=disable_web_page_preview,
                )
            except BadRequest as e:
                if 'too long' in str(e):
                    if input_media is not None:
                        log.info('Caption too long, fallback to text: %s', e)
                        assert media
                        input_media = media.as_input(None, None)
                        assert input_media
                        with input_media as im:
                            r = await bot.edit_message_media(im, m.chat.id, resp_msg_id)
                        if text:
                            return await _reply_text(text)
                        return r
                    if as_caption:
                        log.info('Too long for caption, fallback to text: %s', e)
                        assert text
                        return await _reply_text(text)
                raise
        except Exception as e:
            if isinstance(e, TypeError):
                raise

            # Cache expired, remove it first for other coroutines.
            # We don't bypass 'Message is not modified' here, as the user side cannot
            # distinguish whether the message is being updated.
            # This behavior can be overridden by `allow_not_modified`.
            if (
                isinstance(e, BadRequest)
                and 'Message is not modified' in str(e)
                and allow_not_modified
            ):
                log.info('Message not modified: %s -> %s', key, val)
                return False

            db.discard(key)
            fmt = 'Failed to edit response: %s -> %s: %s: %s'
            log.warning(fmt, key, val, type(e).__name__, e)
            return await _do_reply()
