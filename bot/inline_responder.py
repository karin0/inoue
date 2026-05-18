from contextlib import contextmanager
from typing import Awaitable, Callable

from telegram import (
    InlineKeyboardMarkup,
    InlineQueryResult,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
)
from telegram.constants import ChatAction, MessageLimit
from telegram.error import BadRequest

from . import env
from .env import log
from .payload import MediaPayload, MediaPayloadWithInput, payload_has_input
from .responder import Responder, EditHandle
from .message_responder import MessageEditHandle
from .text import escape, html_escape, shorten, truncate_text

type InlineMessageIdFactory = Callable[[Message, InlineQueryResult], Awaitable[str]]


class InlineResponder(Responder):
    '''Wraps a guest message or inline-callback query, producing an
    `inline_message_id` after it is answered and editing in place thereafter.

    Useful for guest messages, which present the UX semantics of normal messages
    on send but must be answered with an `InlineQueryResult`.
    '''

    __slots__ = (
        '_msg',
        '_inline_message_id',
        '_fragments',
        '_reply_markup',
        '_disable_web_page_preview',
        '_cached_idx',
        '_media',
        '_deferred',
        '_dirty',
        '_message_key',
    )

    def __init__(
        self, msg: Message, inline_message_id: str | InlineMessageIdFactory
    ) -> None:
        super().__init__()
        self._msg = msg
        self._inline_message_id: str | InlineMessageIdFactory = inline_message_id
        self._fragments: list[tuple[str, str | None]] = []
        self._reply_markup: InlineKeyboardMarkup | None = None
        self._disable_web_page_preview: bool | None = None
        self._cached_idx: int | None = None
        self._media: MediaPayload | None = None
        self._deferred = False
        self._dirty = False
        self._message_key = None

    def __repr__(self) -> str:
        return f'InlineResponder({self._msg!r}, {self._inline_message_id!r})'

    def get_message(self) -> Message:
        return self._msg

    def get_message_key(self) -> str:
        if self._message_key is None:
            # Cache it to avoid a volatile result.
            if isinstance(mid := self._inline_message_id, str):
                self._message_key = mid
            else:
                m = self._msg
                self._message_key = env.driver.message_key(m.chat_id, m.message_id)

        return self._message_key

    def as_edit_handle(self) -> MessageEditHandle | None:
        if isinstance(mid := self._inline_message_id, str):
            return MessageEditHandle(mid)
        # Editing is prevented until the inline message is emitted.

    async def reply_chat_action(self, action: ChatAction) -> bool:
        log.debug('InlineResponder: ignored reply_chat_action: %s', action)
        return False

    @contextmanager
    def keep_chat_action(self, action: ChatAction):
        log.debug('InlineResponder: ignored keep_chat_action: %s', action)
        yield

    def wait_until[T](self, coro: Awaitable[T]) -> Awaitable[T] | None:
        # A `MediaPayload` without `InputMedia` cannot be edited onto the inline
        # message.
        # When a pending voice is present, the caller (or dispatcher) must defer
        # our emission until the voice is ready.
        if isinstance(self._inline_message_id, str):
            log.warning(
                'InlineResponder: cannot defer with existing inline message: %s',
                self,
            )
            return None

        log.info('InlineResponder: deferred: %s', self)
        self._deferred = True
        return self._flush_after(coro)

    async def _flush_after[T](self, coro: Awaitable[T]) -> T:
        try:
            return await coro
        finally:
            await self.flush()

    async def flush(self) -> None:
        self._deferred = False
        if self._dirty:
            log.info('InlineResponder: flushing deferred: %s', self)
            await self._emit()

    async def _set_media(self, payload: MediaPayload) -> None:
        if isinstance(self._inline_message_id, str):
            if not payload_has_input(payload):
                log.warning(
                    'InlineResponder: inline message sent but InputMedia is unavailable, '
                    'consider using `InlineResponder.wait_until()`: %s',
                    payload,
                )
                return
            log.debug('InlineResponder: set media: %s', payload)
            self._media = payload
        elif (media := await payload.as_cached()) is not None:
            log.debug('InlineResponder: staged media: %s', media)
            self._media = media

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
    ) -> InlineFragmentHandle:
        if self._try_capture(text, parse_mode):
            cached = False

        if reply_markup is not None:
            if self._reply_markup is not None:
                log.warning(
                    'InlineResponder: overriding reply_markup: %s -> %s',
                    self._reply_markup,
                    reply_markup,
                )
            self._reply_markup = reply_markup

        if disable_web_page_preview is not None:
            self._disable_web_page_preview = disable_web_page_preview

        if media is not None:
            if self._media is not None:
                log.warning('InlineResponder: ignored extra media: %r', media)
                media = None
            else:
                await self._set_media(media)

        frag = (text or '', parse_mode if text else None)
        if cached and (idx := self._cached_idx) is not None:
            log.debug(
                'InlineResponder: editing fragment %d: %s -> %s',
                idx,
                self._fragments[idx],
                frag,
            )
            self._fragments[idx] = frag
        else:
            idx = len(self._fragments)
            log.debug('InlineResponder: new fragment %d: %s', idx, frag)
            self._fragments.append(frag)

        if cached:
            self._cached_idx = idx

        if text or media is not None or reply_markup is not None:
            if allow_not_modified:
                try:
                    await self._emit()
                except BadRequest as e:
                    if 'Message is not modified' in str(e):
                        log.info('InlineResponder.reply: message not modified')
                    else:
                        raise
            else:
                await self._emit()

        return InlineFragmentHandle(self, idx)

    async def _emit(self) -> Message | bool:
        if not self._fragments and self._media is None:
            log.info('InlineResponder: nothing to emit')
            return True

        if self._deferred:
            log.debug('InlineResponder: emitting deferred')
            self._dirty = True
            return True

        text, parse_mode = _collapse_fragments(self._fragments, self._media is None)

        log.debug(
            'InlineResponder: emitting: %r %r %r %r',
            text,
            parse_mode,
            self._reply_markup,
            self._media,
        )

        self._dirty = False
        if isinstance(mid := self._inline_message_id, str):
            log.info('InlineResponder: editing inline message: %s', mid)
            handle = MessageEditHandle(mid)
            if (media := self._media) is not None and not payload_has_input(media):
                # We have warned about this in `_set_media`.
                media = None
            return await handle.edit(
                text or None,
                parse_mode,
                self._reply_markup,
                media=media,
                disable_web_page_preview=self._disable_web_page_preview,
            )

        if (media := self._media) is not None and (
            media := await media.as_cached()
        ) is not None:
            result = media.as_inline_result(
                text or None, parse_mode, self._reply_markup
            )
        else:
            result = InlineQueryResultArticle(
                id='noop',
                title=shorten(text) or 'Text',
                input_message_content=InputTextMessageContent(
                    text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=self._disable_web_page_preview,
                ),
                reply_markup=self._reply_markup,
            )
        log.debug('InlineResponder: emitting inline result: %s', result)
        self._inline_message_id = r = await mid(self._msg, result)
        log.info('InlineResponder: emitted inline message: %s', r)
        return True

    async def reply_copy(
        self, from_chat_id: int, message_id: int
    ) -> InlineFragmentHandle:
        staged = await env.driver.stage_message(from_chat_id, message_id)
        text = staged.text or staged.caption or None
        if (cached := MediaPayload.extract(staged)) is not None:
            return await self.reply(text, media=cached)
        # XXX: This drops the original entities.
        return await self.reply(text)

    def reply_forward(
        self, from_chat_id: int, message_id: int
    ) -> Awaitable[InlineFragmentHandle]:
        return self.reply_copy(from_chat_id, message_id)


def _collapse_fragments(
    fragments: list[tuple[str, str | None]],
    is_text_only: bool,
) -> tuple[str, str | None]:
    all_parse_modes = set(p for _, p in fragments)
    if len(all_parse_modes) == 1:
        parse_mode = all_parse_modes.pop()
        text = '\n\n'.join(t for t, _ in fragments if t)
        return text, parse_mode

    all_parse_modes.discard(None)
    if len(all_parse_modes) == 1:
        parse_mode = all_parse_modes.pop()
        escaper = html_escape if parse_mode == 'HTML' else escape
        text = '\n\n'.join(escaper(t) if m is None else t for t, m in fragments if t)
        return text, parse_mode

    # Markdown and HTML mixed: fallback to plain text.
    text = '\n\n'.join(t for t, _ in fragments if t)
    limit = (
        MessageLimit.MAX_TEXT_LENGTH if is_text_only else MessageLimit.CAPTION_LENGTH
    )
    return truncate_text(text, limit), None


class InlineFragmentHandle(EditHandle):
    __slots__ = ('_rs', '_idx')

    def __init__(self, rs: InlineResponder, idx: int):
        self._rs = rs
        self._idx = idx

    async def edit(
        self,
        text: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        media: MediaPayloadWithInput | None = None,
        disable_web_page_preview: bool | None = None,
        allow_not_modified: bool = False,
    ) -> Message | bool:
        r = self._rs
        if text:
            r._fragments[self._idx] = (text, parse_mode)
        if reply_markup is not None:
            r._reply_markup = reply_markup
        if disable_web_page_preview is not None:
            r._disable_web_page_preview = disable_web_page_preview
        if media is not None:
            # This overrides the media for the entire inline message.
            await r._set_media(media)
        return await r._emit()

    def edit_text(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        disable_web_page_preview: bool | None = None,
    ) -> Awaitable[Message | bool]:
        r = self._rs
        r._fragments[self._idx] = (text, parse_mode)
        if reply_markup is not None:
            r._reply_markup = reply_markup
        if disable_web_page_preview is not None:
            r._disable_web_page_preview = disable_web_page_preview
        return r._emit()

    async def edit_reply_markup(
        self,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message | bool:
        r = self._rs
        if reply_markup is not None:
            r._reply_markup = reply_markup
            return await r._emit()
        if r._reply_markup is not None:
            r._reply_markup = None
            return await r._emit()
        return True

    def get_message_key(self) -> str:
        return self._rs.get_message_key()

    def as_responder(self) -> InlineResponder:
        return self._rs
