# ruff: noqa: F401, F403
import logging

from db import db

from env import (
    USER_ID,
    MEDIA_STAGING_CHAT_ID,
    MEDIA_STAGING_MESSAGE_THREAD_ID,
    MAX_TEXT_LENGTH,
    encode_id,
)

log = logging.getLogger('bot')
