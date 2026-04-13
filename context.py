import os
from typing import NamedTuple
from contextvars import ContextVar

ME = os.environ['ME']
ME_LOWER = ME.lower()


class Sender(NamedTuple):
    id: int
    name: str
    is_guest: bool


current_sender: ContextVar[Sender | None] = ContextVar('current_sender', default=None)


def get_sender() -> Sender | None:
    return current_sender.get()


def is_sender_guest() -> bool:
    sender = get_sender()
    return sender.is_guest if sender else True
