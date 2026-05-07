import os

from telegram.constants import MessageLimit

from context import ME_LOWER


def list_env(key: str, sep: str = ',') -> tuple[str, ...]:
    if val := os.environ.get(key):
        return tuple(r for s in val.split(sep) if (r := s.strip()))
    return ()


def load_ids(key: str) -> tuple[int, ...]:
    return tuple(int(x) for x in list_env(key))


USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])
GROUP_ID = int(os.environ['GROUP_ID'])
TODO_ID = int(os.environ['TODO_ID'])

GUEST_USER_IDS = frozenset(load_ids('GUEST_USER_IDS'))
IGNORE_CHAT_IDS = frozenset(load_ids('IGNORE_CHAT_IDS'))

TRUSTED_IDS = frozenset((USER_ID, CHAN_ID, GROUP_ID, TODO_ID, *load_ids('TRUSTED_IDS')))

LOG_THREAD_ID = int(os.environ.get('LOG_THREAD_ID', 0)) or None

DB_FILE = os.environ.get('DB_FILE', ME_LOWER + '.db')
LOCK_FILE = ME_LOWER + '.pid'

MAX_TEXT_LENGTH = MessageLimit.MAX_TEXT_LENGTH
