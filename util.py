import os
import asyncio
import logging

from telegram import Update, Bot
from telegram.constants import MessageLimit


USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])

MAX_TEXT_LENGTH = MessageLimit.MAX_TEXT_LENGTH


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
    return logger


log = _get_logger('inoue')
log2 = logging.getLogger('inoue.notify')


def get_arg(update: Update) -> str:
    s = (update.message or update.edited_message).text
    if not s.startswith('/'):
        return s.strip()
    try:
        return s[s.index(' ') + 1 :].strip()
    except ValueError:
        return ''


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


bot = None


def init_util(b: Bot):
    global bot
    bot = b


class NotifyHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if bot is not None:
            msg = truncate_text(self.format(record))
            asyncio.create_task(bot.send_message(USER_ID, msg))


log2.setLevel(logging.INFO)
log2.addHandler(NotifyHandler())
