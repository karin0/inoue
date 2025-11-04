import os
import asyncio
import logging
import traceback

from contextvars import ContextVar

from telegram import Message, Bot
from telegram.constants import MessageLimit

USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])
BOT_NAME = os.environ['BOT_NAME']

MAX_TEXT_LENGTH = MessageLimit.MAX_TEXT_LENGTH


class NotifyHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        asyncio.create_task(do_notify(self.format(record)))


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

    h = NotifyHandler()
    h.setLevel(logging.WARNING)
    logger.addHandler(h)

    return logger


log = _get_logger('sendai')

bot = None
msg: ContextVar[Message] = ContextVar('msg')


def init_util(b: Bot):
    global bot
    bot = b


def set_msg(m: Message | None):
    if not m or m.chat.id == USER_ID:
        msg.set(m)
    else:
        msg.set(None)
        log.warning('Bad msg set: %s', m)


async def do_notify(text: str, **kwargs):
    if m := msg.get(None):
        try:
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


def get_arg() -> str:
    s = msg.get().text
    if not s.startswith('/'):
        return s.strip()
    try:
        return s[s.index(' ') + 1 :].strip()
    except ValueError:
        return ''


def get_msg_url(msg_id) -> str:
    chat_id = str(CHAN_ID).removeprefix('-100')
    return f'https://t.me/c/{chat_id}/{msg_id}'


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
