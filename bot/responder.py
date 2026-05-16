import asyncio
from contextlib import contextmanager
from typing import Awaitable, Iterator, Protocol, Literal, overload, TYPE_CHECKING

from .env import log
from .app import create_task

if TYPE_CHECKING:
    from telegram import Message, InlineKeyboardMarkup
    from telegram.constants import ChatAction

    from .dispatch import Route
    from .payload import MediaPayload

# A responder is a wrapped `Message` that enforces an edit-after-reply pattern.
# It takes care of default reply parameters, media payload, inline message adaptation,
# text capturing, caching for in-place editing (opt-in), and so on.


class Responder(Protocol):
    __slots__ = ('_text', '_capture_buf')

    def __init__(self):
        self._text: str | None = None
        self._capture_buf: list[tuple[str, str | None]] | None = None

    @overload
    def reply(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        cached: bool = False,
        media: MediaPayload | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: Literal[False] = False,
    ) -> Awaitable[EditHandle]: ...

    @overload
    def reply(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        cached: bool = False,
        media: MediaPayload | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: Literal[True],
    ) -> Awaitable[EditHandle | None]: ...

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
    ) -> Awaitable[EditHandle | None]: ...

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

    def reply_copy(
        self, from_chat_id: int, message_id: int
    ) -> Awaitable[EditHandle | None]: ...

    def reply_forward(
        self, from_chat_id: int, message_id: int
    ) -> Awaitable[EditHandle | None]:
        '''
        Actually a forwarded message cannot have reply_parameters, but we provide
        this for convenience.
        '''
        ...

    def reply_chat_action(self, action: ChatAction) -> Awaitable: ...

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

    def get_message(self) -> Message: ...

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
        p = next((i for i, c in enumerate(s) if c.isspace() or c == '@'), len(s))
        return s[1:p]

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

    @staticmethod
    def create(msg: Message) -> Responder:
        from .message_responder import MessageResponder

        return MessageResponder(msg)

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
        if (buf := self._capture_buf) is not None:
            log.debug('_try_capture: %r %s', self, text)
            if text:
                buf.append((text, parse_mode))
            return True
        return False


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
