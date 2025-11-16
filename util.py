import os
import asyncio
import logging
import traceback

from typing import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from collections import OrderedDict

from telegram import Message, Bot, MessageEntity, InlineKeyboardMarkup, Update
from telegram.constants import MessageLimit
from telegram.helpers import escape_markdown

USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])
BOT_NAME = os.environ['BOT_NAME']

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
        text = self.format(record)
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
    if m and m.chat.id != USER_ID:
        log.warning('Bad use_msg: %s', m)
        m = None

    token = msg.set(m)
    try:
        yield m
    finally:
        msg.reset(token)


async def do_notify(
    text: str, *, message: Message | None = None, revocable: bool = False, **kwargs
):
    if m := message or msg.get(None):
        try:
            if revocable:
                return await reply_text(m, text, **kwargs)
            else:
                return await m.reply_text(text, do_quote=True, **kwargs)
        except Exception as e:
            traceback.print_exc()
            text += f'\nreply_text: {type(e).__name__}: {e}'
            text = truncate_text(text)

    if bot:
        try:
            await bot.send_message(USER_ID, text, **kwargs)
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
        return msg.get()
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


def get_msg_url(msg_id) -> str:
    chat_id = str(CHAN_ID).removeprefix('-100')
    return f'https://t.me/c/{chat_id}/{msg_id}'


_resp_lru = OrderedDict()
_resp_lru_maxsize = 128


async def reply_text(
    update: MessageSource,
    text: str,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    entities: Sequence[MessageEntity] | None = None,
) -> Message:
    m = get_msg(update)

    # Can only be called when `msg_id` is missing in the cache
    async def _do_reply_text():
        resp = await m.reply_text(
            text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            entities=entities,
            do_quote=True,
        )

        while len(_resp_lru) >= _resp_lru_maxsize:
            _resp_lru.popitem(last=False)
        _resp_lru[msg_id] = resp.message_id

        return resp

    msg_id = m.message_id
    resp_msg_id = _resp_lru.get(msg_id)

    if resp_msg_id is None:
        return await _do_reply_text()

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
        _resp_lru.pop(msg_id, None)

        fmt = 'Failed to edit response: %s -> %s: %s: %s'
        log.warning(fmt, msg_id, resp_msg_id, type(e).__name__, e)

        return await _do_reply_text()

    assert isinstance(resp, Message)
    _resp_lru.move_to_end(msg_id)
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
