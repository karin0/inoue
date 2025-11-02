import os
import logging

from telegram import Update


def get_logger(name):
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


log = get_logger('inoue')


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
