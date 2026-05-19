from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

from telegram import (
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
)
from telegram.constants import ChatAction, MessageLimit
from telegram.error import BadRequest

from . import env
from .app import bot
from .env import log
from .message_responder import MessageEditHandle
from .payload import CachedPayload, MediaPayload, MediaPayloadWithInput, payload_has_input
from .responder import EditHandle, Responder
from .text import escape, html_escape, shorten, truncate_text

if TYPE_CHECKING:
    from collections.abc import Awaitable


class Emitter(Protocol):
    __slots__ = ()

    def get_message_key(self, rs: InlineResponder) -> str: ...
    def as_edit_handle(self) -> MessageEditHandle | None: ...
    async def emit(self, rs: InlineResponder) -> bool: ...
    async def extract_media(self, payload: MediaPayload) -> MediaPayload | None: ...


class InlineResponder(Responder):
    '''Wraps a guest message or inline-callback query, producing an
    `inline_message_id` after it is answered and editing in place thereafter.

    Useful for guest messages, which present the UX semantics of normal messages
    on send but must be answered with an `InlineQueryResult`.
    '''

    __slots__ = (
        '_msg',
        '_fragments',
        '_reply_markup',
        '_disable_web_page_preview',
        '_cached_idx',
        '_media',
        '_emitter',
    )

    def __init__(self, msg: Message, inline_message_id: str | None) -> None:
        super().__init__()
        self._msg = msg
        self._fragments: list[tuple[str, str | None]] = []
        self._reply_markup: InlineKeyboardMarkup | None = None
        self._disable_web_page_preview: bool | None = None
        self._cached_idx: int | None = None
        self._media: MediaPayload | None = None
        self._emitter = (
            InlineEmitter(inline_message_id) if inline_message_id is not None else GuestEmitter()
        )

    def __repr__(self) -> str:
        return f'InlineResponder({self._msg!r}, {self._emitter!r})'

    def get_message(self) -> Message:
        return self._msg

    def get_message_key(self) -> str:
        return self._emitter.get_message_key(self)

    def as_edit_handle(self) -> MessageEditHandle | None:
        return self._emitter.as_edit_handle()

    async def reply_chat_action(self, action: ChatAction) -> bool:
        log.debug('InlineResponder: ignored reply_chat_action: %s', action)
        return False

    @contextmanager
    def keep_chat_action(self, action: ChatAction):
        log.debug('InlineResponder: ignored keep_chat_action: %s', action)
        yield

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
                self._media = await self._emitter.extract_media(media)

        frag = (text or '', parse_mode if text else None)
        if cached and (idx := self._cached_idx) is not None:
            log.debug(
                'InlineResponder: editing fragment %d: %s -> %s', idx, self._fragments[idx], frag
            )
            self._fragments[idx] = frag
        else:
            idx = len(self._fragments)
            log.debug('InlineResponder: new fragment %d: %s', idx, frag)
            self._fragments.append(frag)

        if cached:
            self._cached_idx = idx

        if text or media is not None or reply_markup is not None:
            await self.emit(allow_not_modified=allow_not_modified)

        return InlineFragmentHandle(self, idx)

    def _collapse(self) -> tuple[str, str | None]:
        return _collapse_fragments(self._fragments, self._media is None)

    async def emit(self, *, allow_not_modified: bool = False) -> bool:
        if not self._fragments and self._media is None:
            log.info('InlineResponder: nothing to emit')
            return True

        log.debug(
            'InlineResponder: emitting: %r %r %r %r',
            self._emitter,
            self._fragments,
            self._reply_markup,
            self._media,
        )
        if allow_not_modified:
            try:
                return await self._emitter.emit(self)
            except BadRequest as e:
                if 'Message is not modified' in str(e):
                    log.info('InlineResponder.emit: message unmodified')
                    return True
                raise
        return await self._emitter.emit(self)

    async def reply_copy(self, from_chat_id: int, message_id: int) -> InlineFragmentHandle:
        staged = await env.driver.stage_message(from_chat_id, message_id)
        text = staged.text or staged.caption or None
        if (cached := MediaPayload.extract(staged)) is not None:
            return await self.reply(text, media=cached)
        # XXX: This drops the original entities.
        return await self.reply(text)

    def reply_forward(self, from_chat_id: int, message_id: int) -> Awaitable[InlineFragmentHandle]:
        return self.reply_copy(from_chat_id, message_id)

    def can_defer(self) -> bool:
        return isinstance(self._emitter, GuestEmitter) and not self._emitter._deferred

    def defer_until[T](self, coro: Awaitable[T]) -> Awaitable[T] | None:
        if isinstance(em := self._emitter, GuestEmitter):
            log.info('InlineResponder: deferred: %s', self)
            em._deferred = True
            return self._flush_after(coro)

    async def _flush_after[T](self, coro: Awaitable[T]) -> T:
        try:
            return await coro
        finally:
            await self.flush()

    async def flush(self) -> bool | None:
        if isinstance(em := self._emitter, GuestEmitter):
            em._deferred = False
            if em._dirty:
                return await em.emit(self)


class InlineEmitter(Emitter):
    __slots__ = ('_inline_message_id',)

    def __init__(self, inline_message_id: str) -> None:
        self._inline_message_id = inline_message_id

    def __repr__(self) -> str:
        return f'InlineResponder({self._inline_message_id!r})'

    def get_message_key(self, rs: InlineResponder) -> str:
        return self._inline_message_id

    def as_edit_handle(self) -> MessageEditHandle | None:
        return MessageEditHandle(self._inline_message_id)

    async def extract_media(self, payload: MediaPayload) -> MediaPayloadWithInput | None:
        if payload_has_input(payload):
            log.debug('InlineEmitter: media: %s', payload)
            return payload
        log.warning(
            'InlineEmitter: inline message sent but InputMedia is unavailable, '
            'For guest queries consider using `GuestEmitter.wait_until()`: %s',
            payload,
        )

    async def emit(self, rs: InlineResponder) -> bool:
        mid = self._inline_message_id
        log.info('InlineEmitter: editing inline message: %s', mid)
        handle = MessageEditHandle(mid)
        text, parse_mode = rs._collapse()
        if (media := rs._media) is not None and not payload_has_input(media):
            # We have warned about this in `_set_media`.
            media = None
        r = await handle.edit(
            text or None,
            parse_mode,
            rs._reply_markup,
            media=media,
            disable_web_page_preview=rs._disable_web_page_preview,
        )
        if not isinstance(r, bool):
            raise RuntimeError(f'Not a bool: {r!r}')
        return r


class GuestEmitter(Emitter):
    __slots__ = ('_deferred', '_dirty', '_message_key')

    def __init__(self) -> None:
        self._deferred = False
        self._dirty = False
        self._message_key = None

    def __repr__(self) -> str:
        return f'GuestResponder(deferred={self._deferred}, dirty={self._dirty})'

    def get_message_key(self, rs: InlineResponder) -> str:
        if self._message_key is None:
            # Cache it to avoid a volatile result.
            m = rs._msg
            self._message_key = env.driver.message_key(m.chat_id, m.message_id)

        return self._message_key

    def as_edit_handle(self) -> None:
        return None

    async def extract_media(self, payload: MediaPayload) -> CachedPayload | None:
        if (media := await payload.as_cached()) is not None:
            log.debug('GuestEmitter: staged media: %s', media)
            return media

    async def emit(self, rs: InlineResponder) -> bool:
        if self._deferred:
            log.info('GuestEmitter: emitting deferred: %s', self)
            self._dirty = True
            return True

        text, parse_mode = rs._collapse()
        if (media := rs._media) is not None and (media := await media.as_cached()) is not None:
            result = media.as_inline_result(text or None, parse_mode, rs._reply_markup)
        else:
            result = InlineQueryResultArticle(
                id='noop',
                title=shorten(text) or 'Text',
                input_message_content=InputTextMessageContent(
                    text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=rs._disable_web_page_preview,
                ),
                reply_markup=rs._reply_markup,
            )
        log.debug('GuestEmitter: emitting inline result: %s', result)

        if (gid := rs._msg.guest_query_id) is None:
            raise ValueError('No guest_query_id in message: %s', rs._msg)
        r = await bot.answer_guest_query(gid, result)

        log.info('GuestEmitter: emitted inline message: %s', r)
        rs._emitter = InlineEmitter(r.inline_message_id)
        return True


def _collapse_fragments(
    fragments: list[tuple[str, str | None]], is_text_only: bool
) -> tuple[str, str | None]:
    all_parse_modes = {p for _, p in fragments}
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
    limit = MessageLimit.MAX_TEXT_LENGTH if is_text_only else MessageLimit.CAPTION_LENGTH
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
            r._media = await r._emitter.extract_media(media)
        return await r.emit(allow_not_modified=allow_not_modified)

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
        return r.emit()

    async def edit_reply_markup(
        self, reply_markup: InlineKeyboardMarkup | None = None
    ) -> Message | bool:
        r = self._rs
        if reply_markup is not None:
            r._reply_markup = reply_markup
            return await r.emit()
        if r._reply_markup is not None:
            r._reply_markup = None
            return await r.emit()
        return True

    def get_message_key(self) -> str:
        return self._rs.get_message_key()

    def as_responder(self) -> InlineResponder:
        return self._rs
