from io import FileIO
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
    TypeGuard,
    cast,
)

from telegram import (
    Message,
    InlineKeyboardMarkup,
    ReplyParameters,
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

from .app import bot
from .env import log, MEDIA_STAGING_CHAT_ID, MEDIA_STAGING_MESSAGE_THREAD_ID

type Media = Audio | Document | PhotoSize | Sticker | Video | Voice
CACHED_MEDIA_TYPES = (Audio, Document, PhotoSize, Sticker, Video, Voice, str)

type Content = bytes | str | Path | Media
type ContentInput = bytes | str | FileIO
type InputMediaType = type[InputMedia] | None
type CachedPayload = MediaPayload[Media | str]


@contextmanager
def open_content(content: Content) -> Iterator[Any]:
    if isinstance(content, Path):
        with content.open('rb') as fp:
            yield fp
    else:
        yield content


class MediaPayload[T: Content](Protocol):
    __slots__ = ()

    KIND: ClassVar[str]
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType]

    @property
    def content(self) -> T: ...

    def _send(
        self,
        chat_id: int,
        c,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
        reply_parameters: ReplyParameters | None = None,
        direct_messages_topic_id: int | None = None,
    ) -> Awaitable[Message]: ...

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

    async def send(
        self,
        chat_id: int,
        caption: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await self._send(
                chat_id,
                c,
                caption=caption,
                parse_mode=parse_mode,
                message_thread_id=message_thread_id,
                disable_notification=disable_notification,
            )

    async def reply(
        self,
        msg: Message,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        with open_content(self.content) as c:
            return await self._send(
                msg.chat_id,
                c,
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                message_thread_id=(
                    msg.message_thread_id if msg.is_topic_message else None
                ),
                reply_parameters=ReplyParameters(
                    msg.message_id, allow_sending_without_reply=True
                ),
                direct_messages_topic_id=(
                    None
                    if msg.direct_messages_topic is None
                    else msg.direct_messages_topic.topic_id
                ),
            )

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

    async def stage(self, caption: str | None = None) -> MediaPayload[Media] | None:
        from .ctx import get_context

        msg = await self.send(
            MEDIA_STAGING_CHAT_ID,
            message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,
            caption=caption,
            disable_notification=get_context().sender_is_host(),
        )

        if (r := MediaPayload.extract(msg, self.KIND)) is not None:
            return r
        log.error('Failed to stage media: %s', msg)

    @staticmethod
    def extract(msg: Message, kind: str | None = None) -> MediaPayload[Media] | None:
        media: Media | tuple[PhotoSize, ...] | None
        if kind is not None:
            if media := getattr(msg, kind):
                if isinstance(media, (tuple, list)):
                    media = media[-1]
                return ALL_PAYLOAD[kind](media)

        for k, typ in ALL_PAYLOAD.items():
            if media := getattr(msg, k, None):
                if kind is not None:
                    log.info('extract_media: fallback: %s -> %s', kind, k)
                if isinstance(media, (tuple, list)):
                    media = media[-1]
                return typ(media)

    async def as_cached(self) -> CachedPayload | None:
        if isinstance(self.content, CACHED_MEDIA_TYPES):
            # XXX: A str could be a file_id or URL, but we don't use media URLs.
            return cast(CachedPayload, self)
        return await self.stage()

    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> ContextManager[InputMedia] | None:
        return None

    def as_input_now(
        self: MediaPayload[Media | str | bytes],
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> InputMedia | None:
        return None


class MediaPayloadWithInput[T: Content](MediaPayload[T], Protocol):
    __slots__ = ()

    def _as_input(
        self, c, caption: str | None = None, parse_mode: str | None = None
    ) -> InputMedia: ...

    @contextmanager
    def as_input(
        self, caption: str | None = None, parse_mode: str | None = None
    ) -> Iterator[InputMedia]:
        with open_content(self.content) as c:
            yield self._as_input(c, caption, parse_mode)

    def as_input_now(
        self: MediaPayloadWithInput[Media | str | bytes],
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> InputMedia:
        return self._as_input(self.content, caption, parse_mode)


def payload_has_input(payload: MediaPayload) -> TypeGuard[MediaPayloadWithInput]:
    return payload.INPUT_MEDIA_TYPE is not None


# `InlineQueryResultCachedAudio` accepts no `title`.
# `InputMediaVideo` requires `supports_streaming=True`.
# `InputMediaVoice` does not exist.
# `PhotoSize` occurs as a sequence in `Message`.
# `InlineQueryResultCachedSticker` accepts neither `caption` nor `title`.
# `InputMediaSticker` is added in API 10.0 and unsupported by PTB yet.
# `VideoNote` has neither `InputMedia` nor `InlineQueryResult` type.


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class PhotoPayload[T: Content](MediaPayloadWithInput[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]

    KIND: ClassVar[str] = 'photo'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaPhoto

    def _send(
        self,
        chat_id: int,
        c: ContentInput | PhotoSize,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
        reply_parameters: ReplyParameters | None = None,
        direct_messages_topic_id: int | None = None,
    ) -> Awaitable[Message]:
        return bot.send_photo(
            chat_id,
            c,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            message_thread_id=message_thread_id,
            disable_notification=disable_notification,
            reply_parameters=reply_parameters,
            direct_messages_topic_id=direct_messages_topic_id,
        )

    def _as_input(
        self,
        c: ContentInput | PhotoSize,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> InputMediaPhoto:
        return InputMediaPhoto(c, caption=caption, parse_mode=parse_mode)

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
class DocumentPayload[T: Content](MediaPayloadWithInput[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]
    filename: str | None = None
    disable_content_type_detection: bool | None = None

    KIND: ClassVar[str] = 'document'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaDocument

    def _send(
        self,
        chat_id: int,
        c: ContentInput | Document,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
        reply_parameters: ReplyParameters | None = None,
        direct_messages_topic_id: int | None = None,
    ) -> Awaitable[Message]:
        return bot.send_document(
            chat_id,
            c,
            filename=self.filename,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            disable_content_type_detection=self.disable_content_type_detection,
            message_thread_id=message_thread_id,
            disable_notification=disable_notification,
            reply_parameters=reply_parameters,
            direct_messages_topic_id=direct_messages_topic_id,
        )

    def _as_input(
        self,
        c: ContentInput | Document,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> InputMediaDocument:
        return InputMediaDocument(
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
class VideoPayload[T: Content](MediaPayloadWithInput[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]
    duration: int | None = None
    filename: str | None = None
    thumbnail: bytes | None = None
    cover: bytes | None = None

    KIND: ClassVar[str] = 'video'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaVideo

    def _send(
        self,
        chat_id: int,
        c: ContentInput | Video,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
        reply_parameters: ReplyParameters | None = None,
        direct_messages_topic_id: int | None = None,
    ) -> Awaitable[Message]:
        return bot.send_video(
            chat_id,
            c,
            duration=self.duration,
            filename=self.filename,
            thumbnail=self.thumbnail,
            cover=self.cover,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            supports_streaming=True,
            message_thread_id=message_thread_id,
            disable_notification=disable_notification,
            reply_parameters=reply_parameters,
            direct_messages_topic_id=direct_messages_topic_id,
        )

    def _as_input(
        self,
        c: ContentInput | Video,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> InputMediaVideo:
        return InputMediaVideo(
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
class AudioPayload[T: Content](MediaPayloadWithInput[T]):
    content: T  # pyright: ignore[reportIncompatibleMethodOverride]
    duration: int | None = None
    filename: str | None = None
    title: str | None = None
    performer: str | None = None
    thumbnail: bytes | None = None

    KIND: ClassVar[str] = 'audio'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaAudio

    def _send(
        self,
        chat_id: int,
        c: ContentInput | Audio,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
        reply_parameters: ReplyParameters | None = None,
        direct_messages_topic_id: int | None = None,
    ) -> Awaitable[Message]:
        return bot.send_audio(
            chat_id,
            c,
            duration=self.duration,
            filename=self.filename,
            title=self.title,
            performer=self.performer,
            thumbnail=self.thumbnail,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            message_thread_id=message_thread_id,
            disable_notification=disable_notification,
            reply_parameters=reply_parameters,
            direct_messages_topic_id=direct_messages_topic_id,
        )

    def _as_input(
        self,
        c: ContentInput | Audio,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> InputMediaAudio:
        return InputMediaAudio(
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

    def _send(
        self,
        chat_id: int,
        c: ContentInput | Voice,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
        reply_parameters: ReplyParameters | None = None,
        direct_messages_topic_id: int | None = None,
    ) -> Awaitable[Message]:
        return bot.send_voice(
            chat_id,
            c,
            duration=self.duration,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            message_thread_id=message_thread_id,
            disable_notification=disable_notification,
            reply_parameters=reply_parameters,
            direct_messages_topic_id=direct_messages_topic_id,
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

    def _send(
        self,
        chat_id: int,
        c: ContentInput | Sticker,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
        reply_markup: InlineKeyboardMarkup | None = None,
        message_thread_id: int | None = None,
        disable_notification: bool | None = None,
        reply_parameters: ReplyParameters | None = None,
        direct_messages_topic_id: int | None = None,
    ) -> Awaitable[Message]:
        if caption:
            log.debug('StickerPayload: ignoring caption: %r, %r', caption, parse_mode)
        return bot.send_sticker(
            chat_id,
            c,
            reply_markup=reply_markup,
            message_thread_id=message_thread_id,
            disable_notification=disable_notification,
            reply_parameters=reply_parameters,
            direct_messages_topic_id=direct_messages_topic_id,
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
