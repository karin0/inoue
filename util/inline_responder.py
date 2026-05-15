from __future__ import annotations
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

from .log import log
from .app import bot
from .payload import MediaPayload, extract_media
from .responder import Responder, EditHandle, is_captured
from .text import escape, html_escape, shorten, truncate_text
from .env import MEDIA_STAGING_CHAT_ID, MEDIA_STAGING_MESSAGE_THREAD_ID

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
    )

    def __init__(
        self,
        msg: Message,
        inline_message_id: str | InlineMessageIdFactory,
    ) -> None:
        self._msg = msg
        self._inline_message_id: str | InlineMessageIdFactory = inline_message_id
        self._fragments: list[tuple[str, str | None]] = []
        self._reply_markup: InlineKeyboardMarkup | None = None
        self._disable_web_page_preview: bool | None = None
        self._cached_idx: int | None = None
        self._media: tuple[MediaPayload, str] | None = None
        self._deferred = False
        self._dirty = False

    def __repr__(self) -> str:
        return f'InlineResponder({self._msg!r}, {self._inline_message_id!r})'

    def get_message(self) -> Message:
        return self._msg

    async def reply_chat_action(self, action: ChatAction) -> None:
        log.debug('InlineResponder: ignored chat action: %s', action)

    def wait_until(self, coro: Awaitable) -> Awaitable:
        # No `InputMediaVoice` exists, so a pending voice media cannot be edited
        # into the inline message.
        # Caller must defer our emission until the voice is ready.
        if isinstance(self._inline_message_id, str):
            log.warning(
                'InlineResponder: cannot defer after the inline message is emitted: %s',
                self,
            )
            return coro

        log.info('InlineResponder: deferred: %s', self)
        self._deferred = True
        return self._flush_after(coro)

    async def _flush_after(self, coro: Awaitable) -> None:
        try:
            await coro
        finally:
            self._deferred = False
            if self._dirty:
                await self._emit()

    async def _stage_media(self, payload: MediaPayload) -> None:
        if (
            isinstance(self._inline_message_id, str)
            and payload.INPUT_MEDIA_TYPE is None
        ):
            log.warning(
                'InlineResponder: inline message sent but InputMedia is unavailable, '
                'consider using `InlineResponder._defer()`: %s',
                payload,
            )
            return

        content = payload.content
        if isinstance(content, str):
            # XXX: This could be a file_id or URL, but we don't use media URLs.
            media = (payload, content)
        else:
            staged = await payload.send(
                MEDIA_STAGING_CHAT_ID,
                message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,
            )
            media = extract_media(staged, payload.KIND)
            if media is None:
                log.warning('InlineResponder: media unsent: %r', payload)

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
        if is_captured(self._msg, text, parse_mode):
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
                await self._stage_media(media)

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

        if text or media is not None:
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

    async def _emit(self) -> None:
        if not self._fragments and self._media is None:
            log.info('InlineResponder: nothing to emit')
            return

        if self._deferred:
            log.debug('InlineResponder: emitting deferred')
            self._dirty = True
            return

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
            if (
                self._media is not None
                and (input_media := self._media[0].as_input(text or None, parse_mode))
                is not None
            ):
                with input_media as im:
                    await bot.edit_message_media(
                        im, reply_markup=self._reply_markup, inline_message_id=mid
                    )
            else:
                await bot.edit_message_text(
                    text,
                    parse_mode=parse_mode,
                    reply_markup=self._reply_markup,
                    inline_message_id=mid,
                    disable_web_page_preview=self._disable_web_page_preview,
                )
        else:
            if self._media is not None:
                payload, file_id = self._media
                result = payload.as_inline_result(
                    file_id, text or None, parse_mode, self._reply_markup
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

    async def reply_copy(
        self, from_chat_id: int, message_id: int
    ) -> InlineFragmentHandle:
        staged = await bot.forward_message(
            MEDIA_STAGING_CHAT_ID,
            from_chat_id,
            message_id,
            message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,
            disable_notification=True,
        )
        text = staged.text or staged.caption or None
        if (media := extract_media(staged)) is not None:
            return await self.reply(text, media=media[0])
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
        text = '\n\n'.join(t for t, _ in fragments)
        return text, parse_mode

    all_parse_modes.discard(None)
    if len(all_parse_modes) == 1:
        parse_mode = all_parse_modes.pop()
        escaper = html_escape if parse_mode == 'HTML' else escape
        text = '\n\n'.join(escaper(t) if m is None else t for t, m in fragments)
        return text, parse_mode

    # Markdown and HTML mixed: fallback to plain text.
    text = '\n\n'.join(t for t, _ in fragments)
    limit = (
        MessageLimit.MAX_TEXT_LENGTH if is_text_only else MessageLimit.CAPTION_LENGTH
    )
    return truncate_text(text, limit), None


class InlineFragmentHandle(EditHandle):
    __slots__ = ('_rs', '_idx')

    def __init__(self, rs: InlineResponder, idx: int):
        self._rs = rs
        self._idx = idx

    def edit_text(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        disable_web_page_preview: bool | None = None,
    ) -> Awaitable:
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
    ) -> None:
        r = self._rs
        if reply_markup is not None:
            r._reply_markup = reply_markup
            await r._emit()
        elif r._reply_markup is not None:
            r._reply_markup = None
            await r._emit()

    def get_message(self) -> Message:
        return self._rs.get_message()
