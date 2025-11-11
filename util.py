import os
import asyncio
import logging
import traceback
import functools

from typing import Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from telegram import Message, Bot, MessageEntity, InlineKeyboardMarkup, Update
from telegram.constants import MessageLimit
from telegram.helpers import escape_markdown

USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])
BOT_NAME = os.environ['BOT_NAME']

MAX_TEXT_LENGTH = MessageLimit.MAX_TEXT_LENGTH


class NotifyHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self._revocable = False

    def emit(self, record: logging.LogRecord) -> None:
        asyncio.create_task(do_notify(self.format(record), revocable=self._revocable))

    @contextmanager
    def revocable(self):
        self._revocable = True
        try:
            yield
        finally:
            self._revocable = False


_notify_handler = NotifyHandler()
notify_revocable = _notify_handler.revocable


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

    logger.addHandler(_notify_handler)

    return logger


log = _get_logger('sendai')

bot = None

msg: ContextVar[Message | None] = ContextVar('msg')


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


async def do_notify(text: str, *, revocable: bool = False, **kwargs):
    if m := msg.get(None):
        try:
            if revocable:
                return await reply_text(m, text, **kwargs)
            else:
                return await m.reply_text(text, **kwargs)
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
        if m := update.effective_message:
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


# A temporary buffer to populate lru_cache
_the_response: Message | None = None


@functools.lru_cache
def _get_response(msg_id: int) -> Message:
    if not _the_response:
        raise KeyError(msg_id)

    assert _the_response.reply_to_message.message_id == msg_id
    return _the_response


async def reply_text(
    update: MessageSource,
    text: str,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    entities: Sequence[MessageEntity] | None = None,
) -> Message:
    m = get_msg(update)
    msg_id = m.message_id
    try:
        resp = _get_response(msg_id)
    except KeyError:
        resp = await m.reply_text(
            text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            entities=entities,
            do_quote=True,
        )

        # Save resp to cache
        global _the_response
        _the_response = resp
        _get_response(msg_id)
        _the_response = None
    else:
        resp = await resp.edit_text(
            text, parse_mode=parse_mode, reply_markup=reply_markup, entities=entities
        )
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
