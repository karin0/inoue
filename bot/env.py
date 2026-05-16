# ruff: noqa: F401, F403
import logging

from inoue.db import db

from inoue.env import (
    USER_ID,
    MEDIA_STAGING_CHAT_ID,
    MEDIA_STAGING_MESSAGE_THREAD_ID,
    MAX_TEXT_LENGTH,
    encode_id,
)

log = logging.getLogger('bot')
