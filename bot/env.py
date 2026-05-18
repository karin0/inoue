# ruff: noqa: F401, F403
import logging
from typing import Any, Awaitable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Message, Update

    from .payload import MediaPayload
    from .responder import Responder

log = logging.getLogger('bot')


class Driver(Protocol):
    __slots__ = ()

    def __setitem__(self, key: str, value: str) -> None: ...
    def __getitem__(self, key: str) -> str: ...

    def get(self, key: str) -> str | None: ...
    def discard(self, key: str) -> None: ...

    def post_init(self) -> Any: ...
    def post_stop(self) -> Any: ...

    @staticmethod
    def on_error(e: Exception | None) -> None: ...

    @staticmethod
    def message_key(chat_id: int, message_id: int) -> str:
        return f'{chat_id}-{message_id}'

    @staticmethod
    def get_update(rs: Responder | None, /, *, public: bool) -> Update: ...

    @staticmethod
    def stage_media(
        payload: MediaPayload, /, *, caption: str | None
    ) -> Awaitable[Message]: ...

    @staticmethod
    def stage_message(chat_id: int, message_id: int) -> Awaitable[Message]: ...


driver: Driver


def register(d: Driver):
    global driver
    driver = d
