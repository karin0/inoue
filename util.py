import os
import asyncio
import logging
import traceback

from html import escape as html_escape
from typing import Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from telegram import Message, Bot, MessageEntity, InlineKeyboardMarkup, Update
from telegram.constants import MessageLimit
from telegram.helpers import escape_markdown

from db import db
from context import ctx_is_guest

USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])
GROUP_ID = int(os.environ['GROUP_ID'])
BOT_NAME = os.environ['BOT_NAME']
GUEST_USER_IDS = frozenset(
    int(x.strip()) for x in os.environ['GUEST_USER_IDS'].split(',')
)

MAX_TEXT_LENGTH = MessageLimit.MAX_TEXT_LENGTH

msg: ContextVar[Message | None] = ContextVar('msg')
bot: Bot | None = None


class NotifyHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self._revocable = False
        self._suppressed = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._suppressed:
            return
        text = truncate_text(self.format(record))
        # Fetch the context before yielding to async code
        m = msg.get(None)
        asyncio.create_task(do_notify(text, message=m, revocable=self._revocable))

    @contextmanager
    def revocable(self):
        old = self._revocable
        self._revocable = True
        try:
            yield
        finally:
            self._revocable = old

    @contextmanager
    def suppress(self):
        old = self._suppressed
        self._suppressed = True
        try:
            yield
        finally:
            self._suppressed = old


notify = NotifyHandler()


def _get_logger(name):
    if os.environ.get('DEBUG') == '1':
        level = logging.DEBUG
    else:
        level = logging.INFO

    if 'JOURNAL_STREAM' in os.environ:
        fmt = '[%(levelname)s] %(message)s'
    else:
        fmt = '%(asctime)s [%(levelname)s] %(message)s'

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    h = logging.StreamHandler()
    h.setLevel(level)
    h.setFormatter(logging.Formatter(fmt))
    logger.addHandler(h)

    logger.addHandler(notify)

    return logger


log = _get_logger('sendai')


def init_util(b: Bot):
    global bot
    bot = b


@contextmanager
def use_msg(m: Message | None):
    if m and m.chat_id != USER_ID:
        if m.chat_id not in GUEST_USER_IDS and m.chat_id != GROUP_ID:
            log.warning('Bad use_msg: %s', m)
        m = None

    token = msg.set(m)
    try:
        yield m
    finally:
        msg.reset(token)


@contextmanager
def use_is_guest(is_guest: bool):
    if is_guest:
        token_guest = ctx_is_guest.set(True)
    else:
        token_guest = None
    try:
        yield
    finally:
        if token_guest is not None:
            ctx_is_guest.reset(token_guest)


async def do_notify(
    text: str,
    parse_mode: str | None = None,
    *,
    message: Message | None = None,
    revocable: bool = False,
    quiet: bool = False,
    **kwargs,
):
    if m := message or msg.get(None):
        try:
            if revocable:
                return await reply_text(m, text, parse_mode, **kwargs)
            else:
                return await m.reply_text(text, parse_mode, do_quote=True, **kwargs)
        except Exception as e:
            traceback.print_exc()
            text += f'\nreply_text: {type(e).__name__}: {e}'
            text = truncate_text(text)

    if bot:
        try:
            if quiet:
                await bot.send_message(
                    GROUP_ID, text, parse_mode, disable_notification=True, **kwargs
                )
            else:
                await bot.send_message(USER_ID, text, parse_mode, **kwargs)
        except Exception:
            traceback.print_exc()


type MessageSource = Message | Update | None


def get_msg(update: MessageSource) -> Message:
    if isinstance(update, Update):
        # Unlike update.effective_message, channel posts and callback queries
        # are ignored here.
        if m := update.message or update.edited_message:
            return m
    elif update is None:
        if m := msg.get():
            return m
    else:
        return update

    raise ValueError('No message')


def get_msg_arg(update: MessageSource) -> tuple[Message, str]:
    m = get_msg(update)
    s = m.text

    if not s.startswith('/'):
        return m, s.strip()
    try:
        return m, s[s.index(' ') + 1 :].strip()
    except ValueError:
        return m, ''


def get_msg_url(msg_id, chat_id=None) -> str:
    if chat_id is None:
        chat_id = CHAN_ID
    chat_id = str(chat_id).removeprefix('-100')
    return f'https://t.me/c/{chat_id}/{msg_id}'


def get_deep_link_url(arg: str) -> str:
    return f'https://t.me/{BOT_NAME}?start={arg}'


def encode_chat_id(m: Message, default: str = 'u') -> str:
    chat_id = m.chat_id
    if chat_id == USER_ID:
        return default
    if chat_id == CHAN_ID:
        return 'c'
    if chat_id == GROUP_ID:
        return 'g'
    if chat_id in GUEST_USER_IDS:
        log.warning('reply_text: guest chat: %s', m)
        return f'G{chat_id}'
    raise ValueError(f'Bad chat for {m}')


def is_id_trusted(sender_id: int) -> bool:
    return sender_id in (USER_ID, CHAN_ID, GROUP_ID)


# Note: the return value could be `None` if `allow_not_modified` is set.
async def reply_text(
    update: MessageSource,
    text: str,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    *,
    entities: Sequence[MessageEntity] | None = None,
    allow_not_modified: bool = False,
) -> Message | None:
    m = get_msg(update)
    chat_kind = encode_chat_id(m)
    if chat_kind == 'c':
        log.warning('reply_text: channel chat: %s', m)
    key = f'{chat_kind}-{m.message_id}'

    # Can only be called when `key` is missing in the cache.
    async def _do_reply_text():
        resp = await m.reply_text(
            text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            entities=entities,
            do_quote=True,
        )

        val = str(resp.message_id)
        db[key] = val
        log.debug('Sending new response: %s -> %s', key, val)
        return resp

    if not (resp_msg_id := db.get(key)):
        return await _do_reply_text()

    resp_msg_id = int(resp_msg_id)
    log.debug('Editing cached response: %s -> %s', key, resp_msg_id)

    try:
        resp = await bot.edit_message_text(
            text,
            m.chat.id,
            resp_msg_id,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            entities=entities,
        )
    except Exception as e:
        # Cache expired, remove it first for other coroutines.
        # We don't bypass 'Message is not modified' here, as the user side cannot
        # distinguish whether the message is being updated.
        # This behavior can be overridden by `allow_not_modified`.
        e = str(e)
        if 'Message is not modified' not in e:
            del db[key]
            fmt = 'Failed to edit response: %s -> %s: %s: %s'
            log.warning(fmt, key, resp_msg_id, type(e).__name__, e)
        elif allow_not_modified:
            log.info('Message not modified: %s -> %s', key, resp_msg_id)
            return None
        else:
            del db[key]

        return await _do_reply_text()

    assert isinstance(resp, Message)
    return resp


def shorten(s: str | None) -> str:
    if s is None:
        return 'None'
    s = s.strip().replace('\n', ' ').replace('\r', ' ')
    if len(s) > 30:
        return s[:30] + '...'
    return s


def truncate_text(s: str) -> str:
    s = s.strip()
    if len(s) > MAX_TEXT_LENGTH:
        s = s[: MAX_TEXT_LENGTH - 12] + '\n[truncated]'
    return s


def escape(s: str) -> str:
    return escape_markdown(s, version=2)


# This truncates the text by default, trimming any leading and trailing spaces.
def pre_block(s: str, *, do_truncate: bool = True) -> tuple[str, str | None]:
    if len(s) + 8 <= MAX_TEXT_LENGTH:
        text = '```\n' + escape(s) + '\n```'
        if len(text) <= MAX_TEXT_LENGTH:
            return text, 'MarkdownV2'

    if do_truncate:
        s = truncate_text(s)
    return s, None


def blockquote_block(s: str) -> tuple[str, str | None]:
    if len(s) + 36 <= MAX_TEXT_LENGTH:
        text = '<blockquote expandable>' + html_escape(s) + '</blockquote>'
        if len(text) <= MAX_TEXT_LENGTH:
            return text, 'HTML'
    return truncate_text(s), None
