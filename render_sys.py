# For `eval`.
import os
import sys

import logging
import functools
import subprocess
from typing import Callable, Concatenate
from render_core import Box

from util import do_notify, log


class _Syscall(Box):
    def __init__(self) -> None:
        s = os.environ.pop('INOUE_SYSCALL_SECRET_UNSAFE', None)
        if s is None:
            import secrets

            s = secrets.token_urlsafe(16)
            del secrets
        self._secret = s

    # Underscore to avoid being injected into rendering engine.
    async def _init(self) -> None:
        if log.isEnabledFor(logging.DEBUG):
            log.debug('Syscall: %s', self._secret)
        else:
            await do_notify('Syscall: ' + self._secret)

    def auth(self, magic: str) -> bool:
        return self._secret == magic

    @staticmethod
    def _auth[**P, T](
        func: Callable[Concatenate['_Syscall', P], T],
    ) -> Callable[Concatenate['_Syscall', str, P], T]:
        @functools.wraps(func)
        def wrapper(self: _Syscall, magic: str, *args: P.args, **kwargs: P.kwargs) -> T:
            if self._secret != magic:
                raise ValueError('invalid magic')
            log.warning('authorized syscall invoked: %s', func.__name__)
            return func(self, *args, **kwargs)

        return wrapper

    @_auth
    def system(self, cmd: str) -> str:
        result = subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT, timeout=0.1
        )
        return result.strip()

    @_auth
    def evil(self, expr: str):
        return eval(expr)


Syscall = _Syscall()
