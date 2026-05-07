from contextvars import ContextVar
from contextlib import contextmanager

from telegram import Message, Update
from telegram.ext import ContextTypes

from context import Context, Sender, current_context

from .env import USER_ID

text_override: ContextVar[str | None] = ContextVar('text_override', default=None)


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


@contextmanager
def use_text_override(text: str):
    token = text_override.set(text)
    try:
        yield
    finally:
        text_override.reset(token)


def get_msg(update: Update) -> Message:
    # Unlike update.effective_message, channel posts and callback queries
    # are ignored here.
    if m := update.message or update.edited_message:
        return m

    raise ValueError('No message')


def get_arg(m: Message) -> str:
    s = text_override.get() or m.text or m.caption or ''

    if not s.startswith('/'):
        return s.strip()

    p = s.find(' ')
    return s[p + 1 :].strip() if p >= 0 else ''
