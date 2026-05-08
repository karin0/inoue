from typing import NamedTuple
from contextvars import ContextVar
from contextlib import contextmanager

from telegram import Message, Update
from telegram.ext import ContextTypes

from .env import USER_ID


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
    if ctx := get_context():
        return ctx.msg


def get_sender() -> Sender | None:
    if ctx := get_context():
        return ctx.sender


def is_sender_guest() -> bool:
    sender = get_sender()
    return sender.is_guest if sender else True


@contextmanager
def use_context(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    m: Message | None,
    sender: Sender | None,
):
    # Only messages from USER_ID are allowed to be set in the context, since it's
    # used for `do_notify` to notify system events.
    if m and m.chat_id != USER_ID:
        m = None

    token = current_context.set(Context(update, ctx, m, sender))
    try:
        yield m
    finally:
        current_context.reset(token)


def get_msg(update: Update) -> Message:
    # Unlike update.effective_message, channel posts and callback queries
    # are ignored here.
    if m := update.message or update.edited_message:
        return m

    raise ValueError('No message')


text_override: ContextVar[str | None] = ContextVar('text_override', default=None)


@contextmanager
def use_text_override(text: str):
    token = text_override.set(text)
    try:
        yield
    finally:
        text_override.reset(token)


def get_arg(m: Message) -> str:
    s = text_override.get() or m.text or m.caption or ''

    if not s.startswith('/'):
        return s.strip()

    p = min(x for x in (s.find(' '), s.find('\n'), len(s)) if x > 0)
    return s[p + 1 :].strip()
