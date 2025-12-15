import os
from contextvars import ContextVar

ME = os.environ['ME']
ME_LOWER = ME.lower()

ctx_is_guest = ContextVar('ctx_is_guest', default=False)


def is_guest() -> bool:
    return ctx_is_guest.get()
