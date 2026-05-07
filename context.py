import os
from typing import NamedTuple
from contextvars import ContextVar

from telegram import Message, Update
from telegram.ext import ContextTypes

ME = os.environ['ME']
ME_LOWER = ME.lower()


class Sender(NamedTuple):
    id: int
    name: str
    is_guest: bool

    def __str__(self) -> str:
        return f'{self.name} ({self.id}{", guest" if self.is_guest else ""})'


class Context(NamedTuple):
    update: Update
    ptb: ContextTypes.DEFAULT_TYPE
    msg: Message | None
    sender: Sender | None


current_context: ContextVar[Context | None] = ContextVar(
    'current_context', default=None
)

get_context = current_context.get


def get_ctx_msg() -> Message | None:
    if ctx := current_context.get():
        return ctx.msg


def get_sender() -> Sender | None:
    if ctx := current_context.get():
        return ctx.sender


def is_sender_guest() -> bool:
    sender = get_sender()
    return sender.is_guest if sender else True
