from __future__ import annotations
from typing import Any, Awaitable, Callable, Concatenate, NamedTuple, TypeGuard, cast
from functools import partial

from telegram import (
    Message,
    Audio,
    Video,
    Document,
    PhotoSize,
    Voice,
    Sticker,
    InlineKeyboardMarkup,
    InlineQueryResult,
    InlineQueryResultArticle,
    InlineQueryResultCachedAudio,
    InlineQueryResultCachedVideo,
    InlineQueryResultCachedPhoto,
    InlineQueryResultCachedDocument,
    InlineQueryResultCachedVoice,
    InlineQueryResultCachedSticker,
    InputTextMessageContent,
    InputMediaAudio,
    InputMediaVideo,
    InputMediaPhoto,
    InputMediaDocument,
)
from telegram.constants import MessageLimit

from .log import log
from .app import bot
from .text import html_escape, escape, shorten, truncate_text
from .env import MEDIA_STAGING_CHAT_ID, MEDIA_STAGING_MESSAGE_THREAD_ID

type InlineMessageIdFactory[T] = Callable[[T, InlineQueryResult], Awaitable[str]]


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
        self._media: RepliedMedia | None = None
        self._deferred = False
        self._dirty = False

    def __getattr__(self, item: str):
        key = item.removeprefix('reply_')
        if len(key) != len(item):
            if key in MEDIA_TYP:
                return partial(self._reply_media, key, getattr(bot, 'send_' + key))
            if key != 'to_message':
                log.error('InlineMessageProxy: unimplemented reply method: %s', item)

        r = getattr(self._msg, item, None)
        log.debug('InlineMessageProxy: getattr: %s -> %r', item, r)
        return r

    async def _flush(self):
        if self._dirty:
            await self._emit(True)

    def _defer(self):
        # XXX: When a voice media comes, we cannot edit the message to include
        # the voice, since there is no `InputMediaVoice`.
        # In that case, the caller must use `_defer()` in advance to defer the
        # emission until the voice is ready, like in `handle_yt_chosen_result`.
        # See `voice.py`.
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
            log.info('InlineMessageProxy: nothing to emit')
            return

        if self._deferred and not force:
            log.debug('InlineMessageProxy: emitting deferred')
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
            log.info('InlineMessageProxy: editing inline message: %s', mid)
            if (
                self._media is not None
                and (input := self._media.as_input()) is not None
            ):
                typ, media = input
                media: Any
                await bot.edit_message_media(
                    typ(media, caption=text or None, parse_mode=parse_mode),
                    reply_markup=self._reply_markup,
                    inline_message_id=mid,
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
                typ, kwargs = self._media.as_cached()
                kwargs: dict[str, Any]
                if supports_caption(typ):
                    result = typ(
                        id='noop',
                        caption=text or None,
                        parse_mode=parse_mode,
                        reply_markup=self._reply_markup,
                        **kwargs,
                    )
                else:
                    if text:
                        log.warning(
                            'InlineMessageProxy: dropped text for media: %r, %r',
                            self._media,
                            text,
                        )
                    result = typ(id='noop', reply_markup=self._reply_markup, **kwargs)
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
        if not text:
            parse_mode = None
        self._fragments.append((text, parse_mode))
        if text or force:
            await self._emit()
        return RepliedMessage(self, idx)

    def reply_text(
        self,
        text: str,
        parse_mode: str | None = None,
        *args,
        reply_markup: InlineKeyboardMarkup | None = None,
        disable_web_page_preview: bool | None = None,
        do_quote: bool = False,
        **kwargs,
    ) -> Awaitable[RepliedMessage]:
        if reply_markup is not None:
            self._set_reply_markup(reply_markup)

        if disable_web_page_preview is not None:
            self._disable_web_page_preview = disable_web_page_preview

        if any(x is not None for x in args) or any(
            x is not None for x in kwargs.values()
        ):
            log.warning('InlineMessageProxy: ignored extra args: %s, %s', args, kwargs)

        return self._push(text, parse_mode)

    async def _reply_media[**P](
        self,
        key: str,
        send: Callable[Concatenate[int, P], Awaitable[Message]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> RepliedMessage:
        kwargs.pop('do_quote', None)
        input = kwargs.get(key)
        if input is None:
            input = args[0]
        log.debug('InlineMessageProxy: reply_%s: %s', key, type(input))

        caption = cast(str, kwargs.get('caption') or '')
        parse_mode = cast(str | None, kwargs.get('parse_mode'))

        if self._media is not None:
            log.warning(
                'InlineMessageProxy: ignored extra media: %s, %s, %s',
                input,
                args,
                kwargs,
            )
            return await self._push(caption, parse_mode)

        if isinstance(input, str):
            media = RepliedMedia.from_file_id(key, input)
        else:
            # Not a `file_id` yet. Send it to get one.
            msg: Message = await send(
                MEDIA_STAGING_CHAT_ID,
                *args,
                **kwargs,
                message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,  # type: ignore
            )
            if (media := RepliedMedia.from_msg(msg, key)) is None:
                log.warning('InlineMessageProxy: media unsent: %r', msg)
                return await self._push(caption, parse_mode)
            log.debug('InlineMessageProxy: media: %r', media)

        self._media = media
        return await self._push(caption, parse_mode, force=True)

    async def reply_copy(
        self, *args, do_quote: bool = False, **kwargs
    ) -> RepliedMessage:
        # `copy_message` does not return a `Message` to retrieve its content.
        msg = await bot.forward_message(
            MEDIA_STAGING_CHAT_ID,
            *args,
            **kwargs,
            message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,
        )

        if (media := RepliedMedia.from_msg(msg)) is not None:
            log.debug('InlineMessageProxy: copied media: %r', msg)
            if self._media is not None:
                log.warning('InlineMessageProxy: ignored media: %r', msg)
            else:
                log.debug('InlineMessageProxy: copied media: %r', msg)
                self._media = media
        else:
            log.debug('InlineMessageProxy: copy: %r', msg)

        # XXX: This drops the original entities.
        return await self._push(msg.text or msg.caption or '', None, force=True)

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


type CachedMediaWithCaption = InlineQueryResultCachedAudio | InlineQueryResultCachedVideo | InlineQueryResultCachedPhoto | InlineQueryResultCachedDocument | InlineQueryResultCachedVoice
type CachedMedia = CachedMediaWithCaption | InlineQueryResultCachedSticker

type InputMedia = InputMediaAudio | InputMediaVideo | InputMediaPhoto | InputMediaDocument
type Media = Audio | Video | Document | PhotoSize | Voice | Sticker


def supports_caption(typ: type[CachedMedia]) -> TypeGuard[type[CachedMediaWithCaption]]:
    return typ is not InlineQueryResultCachedSticker


MDT = {'title': 'noop'}

# `InlineQueryResultCachedAudio` accepts no `title`.
# `InputMediaVideo` requires `supports_streaming=True`.
# `InputMediaVoice` does not exist.
# `PhotoSize` occurs as a sequence in `Message`.
# `InlineQueryResultCachedSticker` accepts neither `caption` nor `title`.

MEDIA_TYP: dict[
    str, tuple[type[CachedMedia], type[InputMedia] | None, dict[str, str]]
] = {
    'voice': (InlineQueryResultCachedVoice, None, MDT),
    'audio': (InlineQueryResultCachedAudio, InputMediaAudio, {}),
    'video': (
        InlineQueryResultCachedVideo,
        cast(type[InputMediaVideo], partial(InputMediaVideo, supports_streaming=True)),
        MDT,
    ),
    'photo': (InlineQueryResultCachedPhoto, InputMediaPhoto, MDT),
    'document': (InlineQueryResultCachedDocument, InputMediaDocument, MDT),
    'sticker': (InlineQueryResultCachedSticker, None, {}),
}


class RepliedMedia(NamedTuple):
    key: str
    media: Media | str
    file_id: str

    @classmethod
    def from_msg(cls, msg: Message, key: str | None = None) -> RepliedMedia | None:
        media: Media | tuple[PhotoSize, ...] | None
        # Do not use `is not None` here, in case the sequence is empty.
        if key is not None and (media := getattr(msg, key)):
            if isinstance(media, (tuple, list)):
                media = media[-1]
            return cls(key, media, media.file_id)

        for k in MEDIA_TYP:
            if media := getattr(msg, k):
                if key is not None:
                    log.debug('RepliedMedia: fallback: %s -> %s', key, k)
                if isinstance(media, (tuple, list)):
                    media = media[-1]
                return cls(k, media, media.file_id)

    @classmethod
    def from_file_id(cls, key: str, file_id: str) -> RepliedMedia:
        return cls(key, file_id, file_id)

    def __repr__(self) -> str:
        return f'{self.key}: {self.media!r})'

    __str__ = __repr__

    def as_cached(self) -> tuple[type[CachedMedia], dict[str, str]]:
        typ, _, d = MEDIA_TYP[self.key]
        return typ, {**d, self.key + '_file_id': self.file_id}

    def as_input(self) -> tuple[type[InputMedia], Media | str] | None:
        if (typ := MEDIA_TYP[self.key][1]) is not None:
            return typ, self.media
        log.warning(
            'RepliedMedia: InputMedia is unavailable for %s, consider using `InlineMessageProxy._defer()`',
            self.key,
        )
