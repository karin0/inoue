from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from typing import (
    Any,
    Awaitable,
    ClassVar,
    ContextManager,
    Iterator,
    Protocol,
    cast,
)

from telegram import (
    Message,
    InlineKeyboardMarkup,
    Audio,
    Document,
    PhotoSize,
    Sticker,
    Video,
    Voice,
    InlineQueryResult,
    InlineQueryResultCachedAudio,
    InlineQueryResultCachedDocument,
    InlineQueryResultCachedPhoto,
    InlineQueryResultCachedSticker,
    InlineQueryResultCachedVideo,
    InlineQueryResultCachedVoice,
    InputMedia,
    InputMediaAudio,
    InputMediaDocument,
    InputMediaPhoto,
    InputMediaVideo,
)

from .log import log
from .app import bot
from .env import MEDIA_STAGING_CHAT_ID, MEDIA_STAGING_MESSAGE_THREAD_ID

type Media = Audio | Document | PhotoSize | Sticker | Video | Voice
type Content = bytes | str | Path | Media
type InputMediaType = type[InputMedia] | None
type CachedPayload = MediaPayload[Media | str]

CACHED_MEDIA_TYPES = (Audio, Document, PhotoSize, Sticker, Video, Voice, str)


@contextmanager
def open_content(content: Content) -> Iterator[Any]:
    if isinstance(content, Path):
        with content.open('rb') as fp:
            yield fp
    else:
        yield content


class MediaPayload[T](Protocol):
    __slots__ = ()

    KIND: ClassVar[str]
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType]

    @property
    def content(self) -> T: ...

    async def reply(
        self,
        msg: Message,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message: ...

    async def send(
        self,
        chat_id: int,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
    ) -> Message: ...

    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> ContextManager[InputMedia] | None: ...

    @staticmethod
    def _as_inline_result(
        file_id: str,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        id: str = 'noop',
        title: str = 'noop',
    ) -> InlineQueryResult: ...

    def as_inline_result(
        self: CachedPayload,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        id: str = 'noop',
        title: str = 'noop',
    ) -> InlineQueryResult:
        if isinstance(content := self.content, str):
            file_id = content
        else:
            file_id = content.file_id
        return self._as_inline_result(
            file_id, caption, parse_mode, reply_markup, id=id, title=title
        )

    def _stage(self, caption: str | None = None) -> Awaitable[Message]:
        from .ctx import get_context

        return self.send(
            MEDIA_STAGING_CHAT_ID,
            message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,
            caption=caption,
            disable_notification=get_context().sender_is_host(),
        )

    async def stage(self, caption: str | None = None) -> MediaPayload[Media] | None:
        msg = await self._stage(caption)
        if (r := MediaPayload.extract(msg, self.KIND)) is not None:
            return r
        log.error('Failed to stage media: %s', msg)

    @staticmethod
    def extract(msg: Message, kind: str | None = None) -> MediaPayload[Media] | None:
        if (r := extract_media(msg, kind)) is not None:
            typ, media = r
            return typ(media)

    async def as_cached(self) -> CachedPayload | None:
        if isinstance(self.content, CACHED_MEDIA_TYPES):
            # XXX: A str could be a file_id or URL, but we don't use media URLs.
            return cast(CachedPayload, self)
        return await self.stage()


# `InlineQueryResultCachedAudio` accepts no `title`.
# `InputMediaVideo` requires `supports_streaming=True`.
# `InputMediaVoice` does not exist.
# `PhotoSize` occurs as a sequence in `Message`.
# `InlineQueryResultCachedSticker` accepts neither `caption` nor `title`.
# `InputMediaSticker` is added in API 10.0 and unsupported by PTB yet.
# `VideoNote` has neither `InputMedia` nor `InlineQueryResult` type.


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class PhotoPayload[T: Content](MediaPayload[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]

    KIND: ClassVar[str] = 'photo'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaPhoto

    async def reply(
        self,
        msg: Message,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await msg.reply_photo(
                c,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                do_quote=True,
                allow_sending_without_reply=True,
            )

    async def send(
        self,
        chat_id: int,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await bot.send_photo(
                chat_id,
                c,
                caption=caption,
                parse_mode=parse_mode,
                message_thread_id=message_thread_id,
                disable_notification=disable_notification,
            )

    @contextmanager
    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> Iterator[InputMediaPhoto]:
        with open_content(self.content) as c:
            yield InputMediaPhoto(c, caption=caption, parse_mode=parse_mode)

    @staticmethod
    def _as_inline_result(
        file_id: str,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        id: str = 'noop',
        title: str = 'noop',
    ) -> InlineQueryResultCachedPhoto:
        return InlineQueryResultCachedPhoto(
            id=id,
            photo_file_id=file_id,
            title=title,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class DocumentPayload[T: Content](MediaPayload[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]
    filename: str | None = None
    disable_content_type_detection: bool | None = None

    KIND: ClassVar[str] = 'document'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaDocument

    async def reply(
        self,
        msg: Message,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await msg.reply_document(
                c,
                filename=self.filename,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                disable_content_type_detection=self.disable_content_type_detection,
                do_quote=True,
                allow_sending_without_reply=True,
            )

    async def send(
        self,
        chat_id: int,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await bot.send_document(
                chat_id,
                c,
                filename=self.filename,
                caption=caption,
                parse_mode=parse_mode,
                disable_content_type_detection=self.disable_content_type_detection,
                message_thread_id=message_thread_id,
                disable_notification=disable_notification,
            )

    @contextmanager
    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> Iterator[InputMediaDocument]:
        with open_content(self.content) as c:
            yield InputMediaDocument(
                c,
                filename=self.filename,
                caption=caption,
                parse_mode=parse_mode,
                disable_content_type_detection=self.disable_content_type_detection,
            )

    @staticmethod
    def _as_inline_result(
        file_id: str,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        id: str = 'noop',
        title: str = 'noop',
    ) -> InlineQueryResultCachedDocument:
        return InlineQueryResultCachedDocument(
            id=id,
            title=title,
            document_file_id=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class VideoPayload[T: Content](MediaPayload[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]
    duration: int | None = None
    filename: str | None = None
    thumbnail: bytes | None = None
    cover: bytes | None = None

    KIND: ClassVar[str] = 'video'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaVideo

    async def reply(
        self,
        msg: Message,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await msg.reply_video(
                c,
                duration=self.duration,
                filename=self.filename,
                thumbnail=self.thumbnail,
                cover=self.cover,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                supports_streaming=True,
                do_quote=True,
                allow_sending_without_reply=True,
            )

    async def send(
        self,
        chat_id: int,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await bot.send_video(
                chat_id,
                c,
                duration=self.duration,
                filename=self.filename,
                thumbnail=self.thumbnail,
                cover=self.cover,
                caption=caption,
                parse_mode=parse_mode,
                supports_streaming=True,
                message_thread_id=message_thread_id,
                disable_notification=disable_notification,
            )

    @contextmanager
    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> Iterator[InputMediaVideo]:
        with open_content(self.content) as c:
            yield InputMediaVideo(
                c,
                duration=self.duration,
                thumbnail=self.thumbnail,
                cover=self.cover,
                caption=caption,
                parse_mode=parse_mode,
                supports_streaming=True,
            )

    @staticmethod
    def _as_inline_result(
        file_id: str,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        id: str = 'noop',
        title: str = 'noop',
    ) -> InlineQueryResultCachedVideo:
        return InlineQueryResultCachedVideo(
            id=id,
            title=title,
            video_file_id=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class AudioPayload[T: Content](MediaPayload[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]
    duration: int | None = None
    filename: str | None = None
    title: str | None = None
    performer: str | None = None
    thumbnail: bytes | None = None

    KIND: ClassVar[str] = 'audio'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaAudio

    async def reply(
        self,
        msg: Message,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await msg.reply_audio(
                c,
                duration=self.duration,
                filename=self.filename,
                title=self.title,
                performer=self.performer,
                thumbnail=self.thumbnail,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                do_quote=True,
                allow_sending_without_reply=True,
            )

    async def send(
        self,
        chat_id: int,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await bot.send_audio(
                chat_id,
                c,
                duration=self.duration,
                filename=self.filename,
                title=self.title,
                performer=self.performer,
                thumbnail=self.thumbnail,
                caption=caption,
                parse_mode=parse_mode,
                message_thread_id=message_thread_id,
                disable_notification=disable_notification,
            )

    @contextmanager
    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> Iterator[InputMediaAudio]:
        with open_content(self.content) as c:
            yield InputMediaAudio(
                c,
                duration=self.duration,
                title=self.title,
                performer=self.performer,
                thumbnail=self.thumbnail,
                caption=caption,
                parse_mode=parse_mode,
            )

    @staticmethod
    def _as_inline_result(
        file_id: str,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        id: str = 'noop',
        title: str = 'noop',
    ) -> InlineQueryResultCachedAudio:
        return InlineQueryResultCachedAudio(
            id=id,
            audio_file_id=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class VoicePayload[T: Content](MediaPayload[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]
    duration: int | None = None

    KIND: ClassVar[str] = 'voice'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = None

    async def reply(
        self,
        msg: Message,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await msg.reply_voice(
                c,
                duration=self.duration,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                do_quote=True,
                allow_sending_without_reply=True,
            )

    async def send(
        self,
        chat_id: int,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await bot.send_voice(
                chat_id,
                c,
                duration=self.duration,
                caption=caption,
                parse_mode=parse_mode,
                message_thread_id=message_thread_id,
                disable_notification=disable_notification,
            )

    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> None:
        return None

    @staticmethod
    def _as_inline_result(
        file_id: str,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        id: str = 'noop',
        title: str = 'noop',
    ) -> InlineQueryResultCachedVoice:
        return InlineQueryResultCachedVoice(
            id=id,
            title=title,
            voice_file_id=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class StickerPayload[T: Content](MediaPayload[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]

    KIND: ClassVar[str] = 'sticker'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = None

    async def reply(
        self,
        msg: Message,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await msg.reply_sticker(
                c,
                reply_markup=reply_markup,
                do_quote=True,
                allow_sending_without_reply=True,
            )

    async def send(
        self,
        chat_id: int,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await bot.send_sticker(
                chat_id,
                c,
                message_thread_id=message_thread_id,
                disable_notification=disable_notification,
            )

    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> None:
        return None

    @staticmethod
    def _as_inline_result(
        file_id: str,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        *,
        id: str = 'noop',
        title: str = 'noop',
    ) -> InlineQueryResultCachedSticker:
        if caption:
            log.warning('StickerPayload: ignoring caption %r', caption)
        return InlineQueryResultCachedSticker(
            id=id, sticker_file_id=file_id, reply_markup=reply_markup
        )


type MediaPayloadType = VoicePayload | AudioPayload | VideoPayload | DocumentPayload | StickerPayload | PhotoPayload

ALL_PAYLOAD: dict[str, type[MediaPayloadType]] = {
    typ.KIND: typ
    for typ in (
        PhotoPayload,
        DocumentPayload,
        VideoPayload,
        AudioPayload,
        VoicePayload,
        StickerPayload,
    )
}


def extract_media(
    msg: Message, kind: str | None = None
) -> tuple[type[MediaPayloadType], Media] | None:
    media: Media | tuple[PhotoSize, ...] | None
    if kind is not None:
        if media := getattr(msg, kind):
            if isinstance(media, (tuple, list)):
                media = media[-1]
            return ALL_PAYLOAD[kind], media

    for k, typ in ALL_PAYLOAD.items():
        if media := getattr(msg, k, None):
            if kind is not None:
                log.info('extract_payload: fallback: %s -> %s', kind, k)
            if isinstance(media, (tuple, list)):
                media = media[-1]
            return typ, media
