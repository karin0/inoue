import asyncio
from typing import Awaitable

from telegram import Message, Update
from telegram.error import NetworkError

from bot import register, bot, Driver, Responder, MediaPayload

from .log import log, is_debug, notify, do_notify
from .ctx import get_context
from .store import DataStore
from .env import (
    ME,
    USER_ID,
    MEDIA_STAGING_CHAT_ID,
    MEDIA_STAGING_MESSAGE_THREAD_ID,
    encode_id,
)


class DriverImpl(DataStore, Driver):
    __slots__ = ()

    def post_init(self):
        from .commands import set_commands, stats

        self.connect()
        log.info('%s initiated: %s', ME, bot.bot)
        if is_debug:
            return set_commands(bot)
        return asyncio.gather(
            set_commands(bot), do_notify(*stats(bot.bot, f'{ME} initiated'))
        )

    def post_stop(self):
        self.close()
        log.info('%s stopped.', ME)

    @staticmethod
    def on_error(e: Exception | None):
        if isinstance(e, NetworkError):
            with notify.suppress():
                log.error('Network error: %s: %s', type(e).__name__, e)
        elif isinstance(e, Exception):
            log.error('Exception: %s: %s', type(e).__name__, e, exc_info=e)
        else:
            log.error('Unknown error: %s', e)

    @staticmethod
    def message_key(chat_id: int, message_id: int) -> str:
        return f'{encode_id(chat_id)}-{message_id}'

    @staticmethod
    def get_update(rs: Responder | None, /, *, public: bool) -> Update:
        update = get_context().update
        if not (
            public or ((u := update.effective_user) is not None and u.id == USER_ID)
        ):
            raise PermissionError('Unauthorized')
        return update

    @staticmethod
    def stage_media(
        payload: MediaPayload, /, *, caption: str | None
    ) -> Awaitable[Message]:
        return payload.send(
            MEDIA_STAGING_CHAT_ID,
            message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,
            caption=caption,
            disable_notification=get_context().sender_is_host(),
        )

    @staticmethod
    def stage_message(chat_id: int, message_id: int) -> Awaitable[Message]:
        return bot.forward_message(
            MEDIA_STAGING_CHAT_ID,
            chat_id,
            message_id,
            message_thread_id=MEDIA_STAGING_MESSAGE_THREAD_ID,
            disable_notification=get_context().sender_is_host(),
        )


driver = DriverImpl()
register(driver)
