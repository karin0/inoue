from __future__ import annotations

import asyncio
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Awaitable, Protocol

from telegram import Message, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.error import BadRequest

from .log import log
from .app import bot, create_task
from .env import CHAN_ID, GROUP_ID, USER_ID
from .payload import MediaPayload

from db import db


class Responder(Protocol):
    __slots__ = ()

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
    ) -> EditHandle | None: ...

    def reply_cached(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        media: MediaPayload | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Awaitable[EditHandle | None]:
        return self.reply(
            text,
            parse_mode,
            reply_markup,
            cached=True,
            media=media,
            disable_web_page_preview=disable_web_page_preview,
            allow_not_modified=allow_not_modified,
        )

    async def reply_copy(
        self, from_chat_id: int, message_id: int
    ) -> EditHandle | None: ...

    def reply_chat_action(self, action: ChatAction) -> Awaitable: ...

    def get_message(self) -> Message: ...

    def get_text(self) -> str:
        from . import get_text

        return get_text(self.get_message())

    async def _keep_action(self, action: ChatAction):
        try:
            while True:
                await self.reply_chat_action(action)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    @contextmanager
    def keep_chat_action(self, action: ChatAction):
        '''Re-send chat action every 4s so it stays visible during long operations.'''
        task = create_task(self._keep_action(action))
        try:
            yield task
        finally:
            task.cancel()


class EditHandle(Protocol):
    __slots__ = ()

    def edit_text(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        disable_web_page_preview: bool | None = None,
    ) -> Awaitable: ...

    def edit_reply_markup(
        self,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Awaitable: ...

    def get_message(self) -> Message | None:
        return None


class MessageEditHandle(EditHandle):
    __slots__ = ('chat_id', 'message_id', 'as_caption', 'message')

    def __init__(
        self,
        chat_id: int,
        message_id: int,
        as_caption: bool = False,
        message: Message | bool | None = None,
    ):
        self.chat_id = chat_id
        self.message_id = message_id
        self.as_caption = as_caption
        self.message = message if isinstance(message, Message) else None

    @classmethod
    def from_message(cls, msg: Message, as_caption: bool = False) -> MessageEditHandle:
        return cls(msg.chat_id, msg.message_id, as_caption)

    def edit_text(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        disable_web_page_preview: bool | None = None,
    ) -> Awaitable:
        if self.as_caption:
            return bot.edit_message_caption(
                self.chat_id,
                self.message_id,
                caption=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        return bot.edit_message_text(
            text,
            self.chat_id,
            self.message_id,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            disable_web_page_preview=disable_web_page_preview,
        )

    def edit_reply_markup(
        self,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Awaitable:
        return bot.edit_message_reply_markup(
            self.chat_id, self.message_id, reply_markup=reply_markup
        )

    def get_message(self) -> Message | None:
        return self.message


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


class MessageResponder(Responder):
    __slots__ = ('msg',)

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
    ) -> EditHandle | None:
        m = self.msg
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
        async def _do_reply(save: bool = True) -> MessageEditHandle:
            if as_caption := media is not None:
                try:
                    resp = await media.reply(m, text, parse_mode, reply_markup)
                except BadRequest as e:
                    if 'too long' in str(e):
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
            return MessageEditHandle.from_message(resp, as_caption)

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
                        resp = await bot.edit_message_media(
                            im,
                            m.chat.id,
                            resp_msg_id,
                            reply_markup=reply_markup,
                        )
                    return MessageEditHandle(m.chat.id, resp_msg_id, True, resp)
                if not text:
                    raise TypeError('Either text or media must be provided')
                if as_caption:
                    resp = await bot.edit_message_caption(
                        m.chat.id,
                        resp_msg_id,
                        caption=text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                else:
                    resp = await bot.edit_message_text(
                        text,
                        m.chat.id,
                        resp_msg_id,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_web_page_preview=disable_web_page_preview,
                    )
                return MessageEditHandle(m.chat.id, resp_msg_id, as_caption, resp)
            except BadRequest as e:
                if 'too long' in str(e):
                    if input_media is not None:
                        log.info('Caption too long, fallback to text: %s', e)
                        assert media
                        input_media = media.as_input(None, None)
                        assert input_media
                        with input_media as im:
                            resp = await bot.edit_message_media(
                                im, m.chat.id, resp_msg_id
                            )
                        if not text:
                            return MessageEditHandle(m.chat.id, resp_msg_id, True, resp)
                        resp = await _reply_text(text)
                        db[key] = str(resp.message_id)
                        return MessageEditHandle.from_message(resp)
                    if as_caption:
                        log.info('Too long for caption, fallback to text: %s', e)
                        assert text
                        resp = await _reply_text(text)
                        db[key] = str(resp.message_id)
                        return MessageEditHandle.from_message(resp)
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
                return None

            db.discard(key)
            log.warning(
                'Failed to edit response: %s -> %s: %s: %s',
                key,
                val,
                type(e).__name__,
                e,
            )
            return await _do_reply()

    def reply_chat_action(self, action: ChatAction):
        return self.msg.reply_chat_action(action)

    async def reply_copy(self, from_chat_id: int, message_id: int) -> EditHandle | None:
        copied = await self.msg.reply_copy(
            from_chat_id,
            message_id,
            do_quote=True,
            allow_sending_without_reply=True,
        )
        return MessageEditHandle(self.msg.chat_id, copied.message_id)

    def get_message(self) -> Message:
        return self.msg
