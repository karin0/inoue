import os

from telegram.constants import MessageLimit

ME = os.environ['ME']
ME_LOWER = ME.lower()

USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])
GROUP_ID = int(os.environ['GROUP_ID'])
TODO_ID = int(os.environ.get('TODO_ID', 0))


def encode_id(chat_id: int, default: str = 'u') -> str:
    if chat_id == USER_ID:
        return default
    if chat_id == CHAN_ID:
        return 'c'
    if chat_id == GROUP_ID:
        return 'g'
    return f'G{chat_id}'


def list_env(key: str, sep: str = ',') -> tuple[str, ...]:
    if val := os.environ.get(key):
        return tuple(r for s in val.split(sep) if (r := s.strip()))
    return ()


def load_ids(key: str) -> tuple[int, ...]:
    return tuple(int(x) for x in list_env(key))


GUEST_USER_IDS = frozenset(load_ids('GUEST_USER_IDS'))
IGNORE_CHAT_IDS = frozenset(load_ids('IGNORE_CHAT_IDS'))

TRUSTED_IDS = frozenset(
    x for x in (USER_ID, CHAN_ID, GROUP_ID, TODO_ID, *load_ids('TRUSTED_IDS')) if x
)

LOG_THREAD_ID = int(os.environ.get('LOG_THREAD_ID', 0)) or None

DB_FILE = os.environ.get('DB_FILE', ME_LOWER + '.db')
LOCK_FILE = ME_LOWER + '.pid'

MAX_TEXT_LENGTH = MessageLimit.MAX_TEXT_LENGTH

MEDIA_STAGING_CHAT_ID = int(os.environ['MEDIA_STAGING_CHAT_ID'])
MEDIA_STAGING_MESSAGE_THREAD_ID = (
    int(os.environ.get('MEDIA_STAGING_MESSAGE_THREAD_ID', 0)) or None
)
