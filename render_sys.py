# For `eval`.
import os
import sys

import secrets
import logging
import functools
import subprocess
from typing import Callable, Concatenate
from render_core import Box

from util import do_notify, log, escape, html_escape


class _Syscall(Box):
    # Underscore to avoid being injected into rendering engine.
    async def _init(self) -> None:
        if secret := os.environ.get('INOUE_SYSCALL_SECRET_UNSAFE'):
            self._secret = secret
        else:
            self._secret = secrets.token_urlsafe(16)

        if log.isEnabledFor(logging.DEBUG):
            log.debug('Syscall: %s', self._secret)
        else:
            await do_notify('Syscall: ' + self._secret)

    def auth(self, magic: str) -> bool:
        return secrets.compare_digest(self._secret, magic)

    @staticmethod
    def _auth[**P, T](
        func: Callable[Concatenate['_Syscall', P], T],
    ) -> Callable[Concatenate['_Syscall', str, P], T]:
        @functools.wraps(func)
        def wrapper(self: _Syscall, magic: str, *args: P.args, **kwargs: P.kwargs) -> T:
            if not self.auth(magic):
                raise ValueError('invalid magic')
            return func(self, *args, **kwargs)

        return wrapper

    @_auth
    def system(self, cmd: str) -> str:
        result = subprocess.check_output(
            cmd,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=0.1,
            env={'LANG': 'C', 'LC_ALL': 'C'},
        )
        return result.strip()

    @_auth
    def eval(self, expr: str):
        return eval(expr)

    @staticmethod
    def uname() -> str:
        r = os.uname()
        return f'{r.sysname} {r.nodename} {r.release} {r.version} {r.machine}'

    @staticmethod
    def escape(text) -> str:
        return escape(str(text))

    @staticmethod
    def html_escape(text) -> str:
        return html_escape(str(text))


Syscall = _Syscall()
