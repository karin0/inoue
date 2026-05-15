from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, ClassVar, ContextManager, Iterator, Protocol

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

type Media = Audio | Document | PhotoSize | Sticker | Video | Voice
type Content = bytes | str | Path | Media
type InputMediaType = type[InputMedia] | None


@contextmanager
def open_content(content: Content) -> Iterator[Any]:
    if isinstance(content, Path):
        with content.open('rb') as fp:
            yield fp
    else:
        yield content


class MediaPayload(Protocol):
    __slots__ = ()

    KIND: ClassVar[str]
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType]
    content: Content

    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
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
        self, caption: str | None, parse_mode: str | None
    ) -> ContextManager[InputMedia] | None: ...

    def as_inline_result(
        self,
        file_id: str,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> InlineQueryResult: ...


type MediaPayloadType = VoicePayload | AudioPayload | VideoPayload | DocumentPayload | StickerPayload | PhotoPayload

ALL_PAYLOAD: dict[str, type[MediaPayloadType]]


def extract_media(
    msg: Message, kind: str | None = None
) -> tuple[MediaPayload, str] | None:
    media: Media | tuple[PhotoSize, ...] | None
    if kind is not None:
        if media := getattr(msg, kind):
            if isinstance(media, (tuple, list)):
                media = media[-1]
            return ALL_PAYLOAD[kind](media), media.file_id

    for k, typ in ALL_PAYLOAD.items():
        if media := getattr(msg, k, None):
            if kind is not None:
                log.info('extract_payload: fallback: %s -> %s', kind, k)
            if isinstance(media, (tuple, list)):
                media = media[-1]
            return typ(media), media.file_id


# `InlineQueryResultCachedAudio` accepts no `title`.
# `InputMediaVideo` requires `supports_streaming=True`.
# `InputMediaVoice` does not exist.
# `PhotoSize` occurs as a sequence in `Message`.
# `InlineQueryResultCachedSticker` accepts neither `caption` nor `title`.
# `InputMediaSticker` is added in API 10.0 and unsupported by PTB yet.
# `VideoNote` has neither `InputMedia` nor `InlineQueryResult` type.


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class PhotoPayload(MediaPayload):
    content: Content

    KIND: ClassVar[str] = 'photo'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaPhoto

    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
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
        self, caption: str | None, parse_mode: str | None
    ) -> Iterator[InputMediaPhoto]:
        with open_content(self.content) as c:
            yield InputMediaPhoto(c, caption=caption, parse_mode=parse_mode)

    def as_inline_result(
        self,
        file_id: str,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> InlineQueryResultCachedPhoto:
        return InlineQueryResultCachedPhoto(
            id='noop',
            photo_file_id=file_id,
            title='noop',
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class DocumentPayload(MediaPayload):
    content: Content
    filename: str | None = None
    disable_content_type_detection: bool | None = None

    KIND: ClassVar[str] = 'document'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaDocument

    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
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
        self, caption: str | None, parse_mode: str | None
    ) -> Iterator[InputMediaDocument]:
        with open_content(self.content) as c:
            yield InputMediaDocument(
                c,
                filename=self.filename,
                caption=caption,
                parse_mode=parse_mode,
                disable_content_type_detection=self.disable_content_type_detection,
            )

    def as_inline_result(
        self,
        file_id: str,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> InlineQueryResultCachedDocument:
        return InlineQueryResultCachedDocument(
            id='noop',
            title='noop',
            document_file_id=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class VideoPayload(MediaPayload):
    content: Content
    duration: int | None = None
    filename: str | None = None
    thumbnail: bytes | None = None
    cover: bytes | None = None

    KIND: ClassVar[str] = 'video'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = InputMediaVideo

    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
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
        self, caption: str | None, parse_mode: str | None
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

    def as_inline_result(
        self,
        file_id: str,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> InlineQueryResultCachedVideo:
        return InlineQueryResultCachedVideo(
            id='noop',
            title='noop',
            video_file_id=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class AudioPayload(MediaPayload):
    content: Content
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
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
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
        self, caption: str | None, parse_mode: str | None
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

    def as_inline_result(
        self,
        file_id: str,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> InlineQueryResultCachedAudio:
        return InlineQueryResultCachedAudio(
            id='noop',
            audio_file_id=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class VoicePayload(MediaPayload):
    content: Content
    duration: int | None = None

    KIND: ClassVar[str] = 'voice'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = None

    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
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

    def as_input(self, caption: str | None, parse_mode: str | None) -> None:
        return None

    def as_inline_result(
        self,
        file_id: str,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> InlineQueryResultCachedVoice:
        return InlineQueryResultCachedVoice(
            id='noop',
            title='noop',
            voice_file_id=file_id,
            caption=caption,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class StickerPayload(MediaPayload):
    content: Content

    KIND: ClassVar[str] = 'sticker'
    INPUT_MEDIA_TYPE: ClassVar[InputMediaType] = None

    async def reply(
        self,
        msg: Message,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
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

    def as_input(self, caption: str | None, parse_mode: str | None) -> None:
        return None

    def as_inline_result(
        self,
        file_id: str,
        caption: str | None,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> InlineQueryResultCachedSticker:
        if caption:
            log.warning('StickerPayload: ignoring caption %r', caption)
        return InlineQueryResultCachedSticker(
            id='noop',
            sticker_file_id=file_id,
            reply_markup=reply_markup,
        )


ALL_PAYLOAD = {
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
