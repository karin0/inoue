from typing import NamedTuple
from contextvars import ContextVar
from contextlib import contextmanager

from telegram import Message, Update
from telegram.ext import ContextTypes

from .env import USER_ID
from .responder import Responder, MessageResponder


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


current_context: ContextVar[Context | None] = ContextVar(
    'current_context', default=None
)


def get_context() -> Context:
    if (ctx := current_context.get()) is not None:
        return ctx
    raise RuntimeError('No context')


def get_ctx_sender() -> Sender | None:
    if (ctx := current_context.get()) is not None:
        return ctx.sender


def get_ctx_msg() -> Message | None:
    if (ctx := current_context.get()) is not None:
        return ctx.msg


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


msg_override: ContextVar[Message | None] = ContextVar('msg_override', default=None)
text_override: ContextVar[str | None] = ContextVar('text_override', default=None)
responder_override: ContextVar[Responder | None] = ContextVar(
    'responder_override', default=None
)


@contextmanager
def use_msg_override(m: Message, text: str):
    token = msg_override.set(m)
    try:
        text_token = text_override.set(text)
        try:
            yield
        finally:
            text_override.reset(text_token)
    finally:
        msg_override.reset(token)


@contextmanager
def use_text_override(text: str):
    token = text_override.set(text)
    try:
        yield
    finally:
        text_override.reset(token)


@contextmanager
def use_responder_override(responder: Responder):
    token = responder_override.set(responder)
    try:
        yield responder
    finally:
        responder_override.reset(token)


def get_msg(update: Update) -> Message:
    # Unlike update.effective_message, channel posts and callback queries
    # are ignored here.
    if (
        m := msg_override.get()
        or update.message
        or update.edited_message
        or update.guest_message
    ):
        return m

    raise ValueError('No message')


def get_text(m: Message) -> str:
    if (s := text_override.get()) is not None:
        return s
    return m.text or m.caption or ''


def get_responder(msg: Message) -> Responder:
    if (r := responder_override.get()) is not None:
        return r
    return MessageResponder(msg)


def get_arg(m: Message) -> str:
    s = get_text(m)
    if not s.startswith('/'):
        return s.strip()

    p = min(x for x in (s.find(' '), s.find('\n'), len(s)) if x > 0)
    return s[p + 1 :].strip()
