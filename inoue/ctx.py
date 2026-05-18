from typing import NamedTuple
from contextvars import ContextVar
from contextlib import contextmanager

from telegram import Message, Update
from telegram.constants import ChatType

from bot import Responder
from .env import USER_ID


class Sender(NamedTuple):
    id: int
    name: str
    is_guest: bool

    def __str__(self) -> str:
        return f'{self.name} ({self.id}{", guest" if self.is_guest else ""})'


class Context(NamedTuple):
    update: Update
    rs: Responder | None
    sender: Sender | None

    @property
    def sender_name(self) -> str:
        if (sender := self.sender) is not None:
            return sender.name
        return '<?>'

    def sender_is_guest(self) -> bool:
        if (sender := self.sender) is not None:
            return sender.is_guest
        return False

    def sender_is_host(self) -> bool:
        if (sender := self.sender) is not None:
            return sender.id == USER_ID
        return False


current_context: ContextVar[Context] = ContextVar('current_context')

get_context = current_context.get


def get_ctx_sender() -> Sender | None:
    if (ctx := get_context(None)) is not None:
        return ctx.sender


def get_ctx_rs() -> Responder | None:
    if (ctx := get_context(None)) is not None:
        return ctx.rs


def is_admin(update: Update, msg: Message) -> bool:
    return (
        (chat := update.effective_chat) is not None
        and chat.type == ChatType.PRIVATE
        and chat.id == USER_ID
        and msg.from_user is not None
        and msg.from_user.id == USER_ID
    )


@contextmanager
def use_context(
    update: Update,
    rs: Responder | None,
    sender: Sender | None,
):
    # Only messages from USER_ID are allowed to be set in the context, since it's
    # used for `do_notify` to notify system events.
    if rs is not None and not is_admin(update, rs.get_message()):
        rs = None

    token = current_context.set(Context(update, rs, sender))
    try:
        yield
    finally:
        current_context.reset(token)
