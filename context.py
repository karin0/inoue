from contextvars import ContextVar

ctx_is_guest = ContextVar('ctx_is_guest', default=False)


def is_guest() -> bool:
    return ctx_is_guest.get()
