from __future__ import annotations
from typing import Awaitable, Callable, Type

from telegram import (
    InlineQueryResult,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InlineQueryResultCachedVoice,
    InputTextMessageContent,
)
from telegram.constants import MessageLimit

from .log import log
from .app import bot
from .text import html_escape, escape, shorten, truncate_text
from .env import MEDIA_STAGING_CHAT_ID, MEDIA_STAGING_MESSAGE_THREAD_ID

type InlineMessageIdFactory[T] = Callable[[T, InlineQueryResult], Awaitable[str]]

type MediaResult = InlineQueryResultCachedVoice


class InlineMessageProxy[T]:
    '''
    This wraps around an inline/guest query, which produces an `inline_message_id`
    after it is answered, making it act like a normal message that can be replied to.

    This is useful for handling guest messages, which have the UX semantics of
    normal messages (in the way how they are sent), but have to be responded to
    with a `InlineQueryResult`.
    '''

    __slots__ = (
        '_msg',
        '_fragments',
        '_reply_markup',
        '_disable_web_page_preview',
        '_inline_message_id',
        '_media',
        '_deferred',
        '_dirty',
    )

    def __init__(
        self,
        msg: T,
        inline_message_id: str | InlineMessageIdFactory[T],
    ) -> None:
        self._msg = msg
        self._fragments: list[tuple[str, str | None]] = []
        self._reply_markup: InlineKeyboardMarkup | None = None
        self._disable_web_page_preview: bool | None = None
        self._inline_message_id: str | InlineMessageIdFactory[T] = inline_message_id
        self._media: tuple[Type[MediaResult], str, str] | None = None
        self._deferred = False
        self._dirty = False

    def __getattr__(self, item):
        return getattr(self._msg, item, None)

    async def _finalize(self):
        if self._dirty:
            await self._emit(True)

    def _defer(self):
        if isinstance(self._inline_message_id, str):
            log.warning(
                'InlineMessageProxy: cannot defer after the inline message is emitted: %s',
                self,
            )
        else:
            self._deferred = True
            log.info('InlineMessageProxy: deferred: %s', self)

    async def _emit(self, force: bool = False) -> None:
        if not self._fragments and self._media is None:
            # TODO: handle media
            log.info('InlineMessageProxy: nothing to emit')
            return

        if self._deferred and not force:
            log.info('InlineMessageProxy: emitting deferred')
            self._dirty = True
            return

        all_parse_modes = set(p for _, p in self._fragments)
        if len(all_parse_modes) == 1:
            parse_mode = all_parse_modes.pop()
            text = '\n'.join(t for t, _ in self._fragments)
        else:
            all_parse_modes.discard(None)
            if len(all_parse_modes) == 1:
                parse_mode = all_parse_modes.pop()
                escaper = html_escape if parse_mode == 'HTML' else escape
                text = '\n'.join(
                    escaper(t) if m is None else t for t, m in self._fragments
                )
            else:
                # Markdown and HTML mixed. Fallback to plain text.
                parse_mode = None
                text = '\n'.join(t for t, _ in self._fragments)
                limit = (
                    MessageLimit.MAX_TEXT_LENGTH
                    if self._media is None
                    else MessageLimit.CAPTION_LENGTH
                )
                text = truncate_text(text, limit)

        log.debug(
            'InlineMessageProxy: emitting: %r %r %r %r',
            text,
            parse_mode,
            self._reply_markup,
            self._media,
        )

        self._dirty = False
        if isinstance(mid := self._inline_message_id, str):
            # XXX: When a voice media comes, we cannot edit the text message to
            # include the media. In that case, the caller must use `_defer()`
            # in advance to defer the emission until the media is ready.
            # See `voice.py`.
            # For other media types, we can use `edit_message_media` like in
            # `handle_yt_chosen_result`, which is not implemented yet.
            log.info('InlineMessageProxy: editing inline message: %s', mid)
            await bot.edit_message_text(
                text,
                parse_mode=parse_mode,
                reply_markup=self._reply_markup,
                inline_message_id=mid,
            )
        else:
            log.info('InlineMessageProxy: emitting inline result: %s', mid)
            if self._media is not None:
                typ, field, value = self._media
                result = typ(
                    id='noop',
                    title=shorten(text) or 'Media',
                    caption=text or None,
                    parse_mode=parse_mode,
                    reply_markup=self._reply_markup,
                    **{field: value},  # type: ignore
                )
            else:
                result = InlineQueryResultArticle(
                    id='noop',
                    title=shorten(text) or 'Text',
                    input_message_content=InputTextMessageContent(
                        text, parse_mode=parse_mode
                    ),
                    reply_markup=self._reply_markup,
                )
            log.debug('InlineMessageProxy: inline result: %s', result)
            self._inline_message_id = await mid(self._msg, result)

    def _set_reply_markup(self, reply_markup: InlineKeyboardMarkup | None):
        if reply_markup is not None:
            if self._reply_markup is None:
                self._reply_markup = reply_markup
                return True
            log.warning(
                'InlineMessageProxy: ignored extra reply_markup: %s', reply_markup
            )
        return False

    async def _push(
        self, text: str = '', parse_mode: str | None = None, *, force: bool = False
    ) -> RepliedMessage:
        idx = len(self._fragments)
        self._fragments.append((text, parse_mode))
        if text or force:
            await self._emit(force)
        return RepliedMessage(self, idx)

    async def reply_text(
        self,
        text: str,
        parse_mode: str | None = None,
        *args,
        reply_markup: InlineKeyboardMarkup | None = None,
        disable_web_page_preview: bool | None = None,
        do_quote: bool = False,
        **kwargs,
    ) -> RepliedMessage:
        if reply_markup is not None:
            self._set_reply_markup(reply_markup)

        if disable_web_page_preview is not None:
            self._disable_web_page_preview = disable_web_page_preview

        if any(x is not None for x in args) or any(
            x is not None for x in kwargs.values()
        ):
            log.warning('InlineMessageProxy: ignored extra args: %s, %s', args, kwargs)

        return await self._push(text, parse_mode)

    async def reply_voice(
        self, voice, *args, do_quote: bool = False, **kwargs
    ) -> RepliedMessage:
        if self._media is not None:
            log.warning(
                'InlineMessageProxy: ignored extra voice: %s, %s, %s',
                voice,
                args,
                kwargs,
            )
            return await self._push()

        if isinstance(voice, str):
            file_id = voice
        else:
            msg = await bot.send_voice(
                MEDIA_STAGING_CHAT_ID,
                voice,
                *args,
                **kwargs,
                message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,
            )
            if (voice := msg.voice) is None:
                raise ValueError('Staged voice not sent')
            file_id = voice.file_id

        self._media = (InlineQueryResultCachedVoice, 'voice_file_id', file_id)
        log.debug('InlineMessageProxy: media: %r', self._media)
        return await self._push(
            kwargs.get('caption', ''), kwargs.get('parse_mode'), force=True
        )

    async def reply_chat_action(self, *args, **kwargs):
        log.info('InlineMessageProxy: ignored chat action: %s, %s', args, kwargs)


class RepliedMessage:
    __slots__ = ('_proxy', '_idx')

    def __init__(self, proxy: InlineMessageProxy, idx: int):
        self._proxy = proxy
        self._idx = idx

    async def edit_text(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *args,
        **kwargs,
    ) -> RepliedMessage:
        if args or kwargs:
            log.warning('RepliedMessage: ignored extra args: %s, %s', args, kwargs)

        proxy = self._proxy
        proxy._fragments[self._idx] = (text, parse_mode)
        proxy._set_reply_markup(reply_markup)
        await proxy._emit()
        return self

    async def edit_reply_markup(
        self,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> RepliedMessage:
        proxy = self._proxy
        if reply_markup is not None:
            if proxy._set_reply_markup(reply_markup):
                await proxy._emit()
        elif proxy._reply_markup is not None:
            proxy._reply_markup = None
            await proxy._emit()
        return self
