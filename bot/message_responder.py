from typing import TYPE_CHECKING

from telegram.error import BadRequest

from . import env
from .app import bot
from .env import log
from .payload import MediaPayload, MediaPayloadWithInput, payload_has_input
from .responder import EditHandle, Responder

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from telegram import InlineKeyboardMarkup, Message
    from telegram.constants import ChatAction


async def edit_message(
    chat_id: int | None = None,
    message_id: int | None = None,
    *,
    as_caption: bool = False,
    inline_message_id: str | None = None,
    text: str | None = None,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    media: MediaPayloadWithInput | None = None,
    disable_web_page_preview: bool | None = None,
    allow_not_modified: bool = False,
) -> Message | bool:
    try:
        if media is not None:
            with media.as_input(text, parse_mode) as im:
                return await bot.edit_message_media(
                    im,
                    chat_id=chat_id,
                    message_id=message_id,
                    inline_message_id=inline_message_id,
                    reply_markup=reply_markup,
                )
        if as_caption:
            return await bot.edit_message_caption(
                chat_id=chat_id,
                message_id=message_id,
                inline_message_id=inline_message_id,
                caption=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        if text:
            return await bot.edit_message_text(
                text,
                chat_id=chat_id,
                message_id=message_id,
                inline_message_id=inline_message_id,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                disable_web_page_preview=disable_web_page_preview,
            )
        if reply_markup is not None:
            return await bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=message_id,
                inline_message_id=inline_message_id,
                reply_markup=reply_markup,
            )
        raise TypeError('Any of text, media, as_caption, or reply_markup must be provided')
    except BadRequest as e:
        if allow_not_modified and 'Message is not modified' in str(e):
            log.info('Message not modified: %s', e)
            return False
        raise


class MessageEditHandle(EditHandle):
    __slots__ = ('chat_id', 'message_id', 'as_caption', 'inline_message_id', '_msg')

    def __init__(
        self, id: tuple[int, int] | str, as_caption: bool = False, message: Message | None = None
    ):
        if isinstance(id, str):
            self.inline_message_id = id
            self.chat_id = None
            self.message_id = None
        else:
            self.chat_id, self.message_id = id
            self.inline_message_id = None
        self.as_caption = as_caption
        self._msg = message

    def get_message_key(self) -> str:
        if self.inline_message_id:
            return self.inline_message_id
        return env.driver.message_key(self.chat_id, self.message_id)  # type: ignore[arg-type]

    def edit(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        media: MediaPayloadWithInput | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Awaitable[Message | bool]:
        return edit_message(
            chat_id=self.chat_id,
            message_id=self.message_id,
            inline_message_id=self.inline_message_id,
            as_caption=self.as_caption,
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            media=media,
            disable_web_page_preview=disable_web_page_preview,
            allow_not_modified=allow_not_modified,
        )

    def edit_text(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Awaitable[Message | bool]:
        return self.edit(
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            disable_web_page_preview=disable_web_page_preview,
            allow_not_modified=allow_not_modified,
        )

    def edit_media(
        self,
        media: MediaPayloadWithInput,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Awaitable[Message | bool]:
        with media.as_input(text, parse_mode) as im:
            return bot.edit_message_media(
                im,
                chat_id=self.chat_id,
                message_id=self.message_id,
                inline_message_id=self.inline_message_id,
                reply_markup=reply_markup,
            )

    def edit_reply_markup(
        self, reply_markup: InlineKeyboardMarkup | None = None
    ) -> Awaitable[Message | bool]:
        return bot.edit_message_reply_markup(
            self.chat_id,
            self.message_id,
            inline_message_id=self.inline_message_id,
            reply_markup=reply_markup,
        )

    def as_responder(self) -> MessageResponder | None:
        if self._msg is not None:
            return MessageResponder(self._msg)


class MessageResponder(Responder):
    __slots__ = ('msg',)

    def __init__(self, msg: Message):
        super().__init__()
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
    ) -> MessageEditHandle:
        m = self.msg
        key = env.driver.message_key(m.chat_id, m.message_id)

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
            if media is not None:
                try:
                    resp = await media.reply(m, text, parse_mode, reply_markup)
                except BadRequest as e:
                    if 'too long' in str(e):
                        log.info('_do_reply: too long for caption, fallback to text: %s', e)
                        resp = await media.reply(m, None, None, None)
                        if text:
                            resp = await _reply_text(text)
                    else:
                        raise
            elif text:
                resp = await _reply_text(text)
            else:
                raise TypeError('Either text or media must be provided')

            r = MessageEditHandle.from_message(resp)
            if save:
                val = str(resp.message_id)
                if r.as_caption:
                    val = '@' + val
                env.driver[key] = val
                log.debug('_do_reply: %s -> %s', key, val)
            return r

        if self._try_capture(text, parse_mode) or not cached:
            return await _do_reply(False)

        if not (val := env.driver.get(key)):
            return await _do_reply()

        if media is not None and not payload_has_input(media):
            env.driver.discard(key)
            return await _do_reply()

        if val[0] == '@':
            as_caption = True
            resp_msg_id = int(val[1:])
        else:
            as_caption = False
            resp_msg_id = int(val)

        log.debug('Editing cached response: %s -> %s', key, val)
        r = MessageEditHandle((m.chat_id, resp_msg_id), as_caption)

        try:
            try:
                await r.edit(
                    text=text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    media=media,
                    disable_web_page_preview=disable_web_page_preview,
                    allow_not_modified=allow_not_modified,
                )
            except BadRequest as e:
                if 'too long' in str(e):
                    if media is not None:
                        log.info('Caption too long, fallback to text: %s', e)
                        resp = await r.edit_media(media, text, parse_mode, reply_markup)
                        if text:
                            resp = await _reply_text(text)
                            env.driver[key] = str(resp.message_id)
                            return MessageEditHandle.from_message(resp)
                        return r
                    if r.as_caption:
                        log.info('Too long for caption, fallback to text: %s', e)
                        assert text
                        resp = await _reply_text(text)
                        env.driver[key] = str(resp.message_id)
                        return MessageEditHandle.from_message(resp)
                raise
            else:
                return r
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
                return r

            env.driver.discard(key)
            log.warning('Failed to edit response: %s -> %s: %s: %s', key, val, type(e).__name__, e)
            return await _do_reply()

    def reply_chat_action(self, action: ChatAction) -> Awaitable[bool]:
        return self.msg.reply_chat_action(action)

    async def reply_copy(self, from_chat_id: int, message_id: int) -> MessageEditHandle:
        copied = await self.msg.reply_copy(
            from_chat_id, message_id, do_quote=True, allow_sending_without_reply=True
        )
        return MessageEditHandle((self.msg.chat_id, copied.message_id), as_caption=False)

    async def reply_forward(self, from_chat_id: int, message_id: int) -> MessageEditHandle:
        msg = await self.msg.get_bot().forward_message(
            self.msg.chat_id, from_chat_id, message_id, message_thread_id=self.msg.message_thread_id
        )
        return MessageEditHandle.from_message(msg)

    def get_message(self) -> Message:
        return self.msg

    def as_edit_handle(self) -> MessageEditHandle | None:
        if (u := self.msg.from_user) is not None and u.is_bot:
            return MessageEditHandle.from_message(self.msg)
