from typing import TYPE_CHECKING, Literal, cast

from telegram import InlineKeyboardMarkup, Message
from telegram.error import BadRequest

from . import env
from .app import bot
from .env import log
from .payload import MediaPayload, MediaPayloadWithInput, payload_has_input
from .responder import EditHandle, ReplyMarkup, Responder

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from telegram.constants import ChatAction


class BotEditHandle[R: Message | bool](EditHandle):
    '''The Bot API returns the edited `Message` for a chat message and `True` for
    an inline message, which `R` records per subclass.'''

    __slots__ = ('as_caption',)

    def __init__(self, as_caption: bool) -> None:
        self.as_caption = as_caption

    def _target(self) -> tuple[int | None, int | None, str | None]: ...

    async def edit(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        media: MediaPayloadWithInput | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> R:
        chat_id, message_id, inline_message_id = self._target()
        try:
            if media is not None:
                with media.as_input(text, parse_mode) as im:
                    r = await bot.edit_message_media(
                        im,
                        chat_id=chat_id,
                        message_id=message_id,
                        inline_message_id=inline_message_id,
                        reply_markup=reply_markup,
                    )
            elif self.as_caption:
                r = await bot.edit_message_caption(
                    chat_id=chat_id,
                    message_id=message_id,
                    inline_message_id=inline_message_id,
                    caption=text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                )
            elif text:
                r = await bot.edit_message_text(
                    text,
                    chat_id=chat_id,
                    message_id=message_id,
                    inline_message_id=inline_message_id,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    disable_web_page_preview=disable_web_page_preview,
                )
            elif reply_markup is not None:
                r = await bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    inline_message_id=inline_message_id,
                    reply_markup=reply_markup,
                )
            else:
                raise TypeError('Any of text, media, as_caption, or reply_markup must be provided')
        except BadRequest as e:
            if allow_not_modified and 'Message is not modified' in str(e):
                log.info('Message not modified: %s', e)
                return cast('R', False)
            raise
        return cast('R', r)

    def edit_text(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Awaitable[R]:
        return self.edit(
            text=text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            disable_web_page_preview=disable_web_page_preview,
            allow_not_modified=allow_not_modified,
        )

    async def edit_reply_markup(self, reply_markup: InlineKeyboardMarkup | None = None) -> R:
        chat_id, message_id, inline_message_id = self._target()
        return cast(
            'R',
            await bot.edit_message_reply_markup(
                chat_id, message_id, inline_message_id=inline_message_id, reply_markup=reply_markup
            ),
        )


class MessageEditHandle(BotEditHandle[Message | Literal[False]]):
    __slots__ = ('chat_id', 'message_id', '_msg')

    def __init__(
        self,
        chat_id: int,
        message_id: int,
        as_caption: bool = False,
        message: Message | None = None,
    ):
        super().__init__(as_caption)
        self.chat_id = chat_id
        self.message_id = message_id
        self._msg = message

    def _target(self) -> tuple[int, int, None]:
        return self.chat_id, self.message_id, None

    def get_message_key(self) -> str:
        return env.driver.message_key(self.chat_id, self.message_id)

    def __repr__(self) -> str:
        return (
            f'MessageEditHandle({self.get_message_key()!r}, as_caption={self.as_caption}, '
            f'msg={self._msg!r})'
        )

    def as_responder(self) -> MessageResponder | None:
        if self._msg is not None:
            return MessageResponder(self._msg)


class InlineMessageEditHandle(BotEditHandle[bool]):
    __slots__ = ('inline_message_id',)

    def __init__(self, inline_message_id: str):
        super().__init__(as_caption=False)
        self.inline_message_id = inline_message_id

    def _target(self) -> tuple[None, None, str]:
        return None, None, self.inline_message_id

    def get_message_key(self) -> str:
        return self.inline_message_id

    def __repr__(self) -> str:
        return f'InlineMessageEditHandle({self.inline_message_id!r})'


class MessageResponder(Responder):
    __slots__ = ('msg',)

    def __init__(self, msg: Message):
        super().__init__()
        self.msg = msg

    async def reply(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: ReplyMarkup | None = None,
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

        if (media is not None and not payload_has_input(media)) or (
            reply_markup is not None and not isinstance(reply_markup, InlineKeyboardMarkup)
        ):
            env.driver.discard(key)
            return await _do_reply()

        if val[0] == '@':
            as_caption = True
            resp_msg_id = int(val[1:])
        else:
            as_caption = False
            resp_msg_id = int(val)

        log.debug('Editing cached response: %s -> %s', key, val)
        r = MessageEditHandle(m.chat_id, resp_msg_id, as_caption)

        try:
            try:
                edited = await r.edit(
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
                        resp = await r.edit(media=media, reply_markup=reply_markup)
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
                if edited is False:
                    # `edit` swallowed a 'not modified' BadRequest under
                    # `allow_not_modified`. Return the cached handle as-is.
                    return r
                return MessageEditHandle.from_message(edited)
        except Exception as e:
            if isinstance(e, TypeError):
                raise

            # Cache expired, remove it first for other coroutines.
            # 'Message is not modified' without `allow_not_modified` falls through
            # here intentionally, since the user side cannot distinguish whether the
            # message is being updated.
            env.driver.discard(key)
            log.warning('Failed to edit response: %s -> %s: %s: %s', key, val, type(e).__name__, e)
            return await _do_reply()

    def reply_chat_action(self, action: ChatAction) -> Awaitable[bool]:
        return self.msg.reply_chat_action(action)

    async def reply_copy(self, from_chat_id: int, message_id: int) -> MessageEditHandle:
        copied = await self.msg.reply_copy(
            from_chat_id, message_id, do_quote=True, allow_sending_without_reply=True
        )
        return MessageEditHandle(self.msg.chat_id, copied.message_id)

    async def reply_forward(self, from_chat_id: int, message_id: int) -> MessageEditHandle:
        msg = await self.msg.get_bot().forward_message(
            self.msg.chat_id, from_chat_id, message_id, message_thread_id=self.msg.message_thread_id
        )
        return MessageEditHandle.from_message(msg)

    def get_message(self) -> Message:
        return self.msg

    def get_message_key(self) -> str:
        return env.driver.message_key(self.msg.chat_id, self.msg.message_id)

    def as_edit_handle(self) -> MessageEditHandle | None:
        if (u := self.msg.from_user) is not None and u.is_bot:
            return MessageEditHandle.from_message(self.msg)
