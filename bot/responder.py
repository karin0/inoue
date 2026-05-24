import asyncio

from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from telegram import Chat, InlineKeyboardMarkup, Message, Update, User
from telegram.constants import ChatAction, ChatType

from . import env
from .app import bot, create_task
from .env import log

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator

    from .dispatch import Route
    from .message_responder import MessageEditHandle
    from .payload import MediaPayload, MediaPayloadWithInput

# A responder is a wrapped `Message` that enforces an edit-after-reply pattern.
# It takes care of default reply parameters, media payload, inline message adaptation,
# text capturing, caching for in-place editing (opt-in), and so on.


class Responder(Protocol):
    __slots__ = ('_text', '_capture_buf')

    def __init__(self):
        self._text: str | None = None
        self._capture_buf: list[tuple[str, str | None]] | None = None

    def reply(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        cached: bool = False,
        media: MediaPayload | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Awaitable[EditHandle]: ...

    def reply_cached(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        media: MediaPayload | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Awaitable[EditHandle]:
        return self.reply(
            text,
            parse_mode,
            reply_markup,
            cached=True,
            media=media,
            disable_web_page_preview=disable_web_page_preview,
            allow_not_modified=allow_not_modified,
        )

    def reply_copy(self, from_chat_id: int, message_id: int) -> Awaitable[EditHandle]: ...

    def reply_forward(self, from_chat_id: int, message_id: int) -> Awaitable[EditHandle]:
        '''
        Actually a forwarded message cannot have reply_parameters, but we provide
        this for convenience.
        '''
        ...

    def reply_chat_action(self, action: ChatAction) -> Awaitable[bool]: ...

    async def _keep_action(self, action: ChatAction):
        while True:
            if not await self.reply_chat_action(action):
                log.warning('reply_chat_action failed: %s', self)
                return
            await asyncio.sleep(4)

    @contextmanager
    def keep_chat_action(self, action: ChatAction):
        '''Re-send chat action every 4s so it stays visible during long operations.'''
        task = create_task(self._keep_action(action))
        try:
            yield
        finally:
            task.cancel()

    def get_message(self) -> Message: ...

    def as_edit_handle(self) -> MessageEditHandle | None:
        return None

    def get_message_key(self) -> str:
        msg = self.get_message()
        return env.driver.message_key(msg.chat_id, msg.message_id)

    def get_text(self) -> str:
        if self._text is not None:
            return self._text
        msg = self.get_message()
        return msg.text or msg.caption or ''

    def set_text(self, text: str) -> None:
        self._text = text

    def get_cmd(self) -> str:
        s = self.get_text()
        if not s.startswith('/'):
            return ''
        p = next((i for i, c in enumerate(s) if c.isspace()), len(s))
        cmd = s[1:p]
        q = cmd.find('@')
        if q != -1:
            if cmd[q + 1 :] != bot.username:
                # Omit commands for other bots.
                return ''
            cmd = cmd[:q]
        return cmd

    def get_arg(self) -> str:
        s = self.get_text()
        if not s.startswith('/'):
            return s.strip()
        p = next((i for i, c in enumerate(s) if c.isspace()), len(s))
        return s[p + 1 :].strip()

    def get_route(self) -> Route | None:
        from .dispatch import commands

        if cmd := self.get_cmd():
            return commands.get(cmd)

    def dispatch_command(self) -> Awaitable | None:
        if (route := self.get_route()) is not None:
            return route(self)

    def dispatch_callback_query(self, data: str) -> Awaitable | None:
        from .dispatch import dispatch_callback

        return dispatch_callback(self, data)

    def dispatch_start(self) -> Awaitable | None:
        from .dispatch import dispatch_start

        return dispatch_start(self, self.get_arg())

    @contextmanager
    def capture(
        self, buf: list[tuple[str, str | None]] | None = None
    ) -> Iterator[list[tuple[str, str | None]]]:
        if buf is None:
            buf = []
        old = self._capture_buf
        self._capture_buf = buf
        try:
            yield buf
        finally:
            self._capture_buf = old

    def is_captured(self) -> bool:
        return self._capture_buf is not None

    def _try_capture(self, text: str | None, parse_mode: str | None) -> bool:
        '''
        Note: Do not try to edit the reply if the reply is captured, or we will
        mess up the original reply of the capturing context (`/render`).

        Always ignore `cached` when this returns True.
        '''
        if (buf := self._capture_buf) is not None:
            log.debug('_try_capture: %r %s', self, text)
            if text:
                buf.append((text, parse_mode))
            return True
        return False

    @staticmethod
    def create(update: Update) -> Responder | None:
        from .inline_responder import InlineResponder
        from .message_responder import MessageResponder

        if (
            msg := update.message
            or update.edited_message
            or update.channel_post
            or update.edited_channel_post
        ) is not None:
            return MessageResponder(msg)

        if (callback := update.callback_query) is not None:
            if isinstance(msg := callback.message, Message):
                return MessageResponder(msg)

            if mid := callback.inline_message_id:
                if msg is not None:
                    stub = Message(
                        msg.message_id, msg.date, msg.chat, from_user=callback.from_user, text=''
                    )
                else:
                    stub = _stub(callback.from_user)
                return InlineResponder(stub, mid)

            return None

        if (chosen := update.chosen_inline_result) is not None:
            if mid := chosen.inline_message_id:
                return InlineResponder(_stub(chosen.from_user), mid)
            return None

        if (msg := update.guest_message) is not None:
            return InlineResponder(msg, None)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.get_message()})'


def _stub(from_user: User) -> Message:
    return Message(0, datetime.now(), Chat(0, ChatType.SENDER), from_user=from_user, text='')


class EditHandle(Protocol):
    __slots__ = ()

    def edit(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        media: MediaPayloadWithInput | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Awaitable[Message | bool]: ...

    def edit_text(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        disable_web_page_preview: bool | None = None,
    ) -> Awaitable[Message | bool]: ...

    def edit_reply_markup(
        self, reply_markup: InlineKeyboardMarkup | None = None
    ) -> Awaitable[Message | bool]: ...

    def get_message_key(self) -> str | None:
        return None

    def as_responder(self) -> Responder | None:
        return None

    @staticmethod
    def from_message(message: Message) -> MessageEditHandle:
        from .message_responder import MessageEditHandle

        return MessageEditHandle(
            (message.chat_id, message.message_id), as_caption=message.text is None, message=message
        )
