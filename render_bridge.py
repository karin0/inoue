# For `eval`.
import os
import sys

import subprocess
from typing import Callable, Concatenate
from render_core import Box, Context

from util import escape, html_escape, log
from motto import hitokoto


class Signature(Box):
    def __init__(self, token: int):
        self.token = token

    def __repr__(self) -> str:
        return f'<Signature: {self.token}>'


class _Bridge(Box):
    def __repr__(self) -> str:
        return '<Bridge>'

    # Underscore to avoid being injected into rendering engine.
    @staticmethod
    def _auth[**P, T](
        func: Callable[P, T],
    ) -> Callable[Concatenate[Context, P], T]:

        # Not using `wraps()` to keep opaque.
        def wrapper(ctx: Context, *args: P.args, **kwargs: P.kwargs) -> T:
            # See `RenderContext.__init__`.
            if isinstance(signature := ctx.get('_trusted'), Signature):
                log.debug(
                    '_Syscall: authorized %s for %s\n  ctx = %s',
                    signature.token,
                    func.__name__,
                    ctx,
                )
                return func(*args, **kwargs)
            log.warning(
                '_Syscall: unauthorized access to %s\n  ctx = %s',
                func.__name__,
                ctx,
            )
            raise PermissionError('unauthorized')

        return wrapper

    @staticmethod
    @_auth
    def system(cmd: str) -> str:
        result = subprocess.check_output(
            cmd,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=0.1,
            env={'LANG': 'C', 'LC_ALL': 'C'},
        )
        return result.strip()

    @staticmethod
    @_auth
    def eval(expr: str):
        return eval(expr)

    @staticmethod
    @_auth
    def uname() -> str:
        r = os.uname()
        return f'{r.sysname} {r.nodename} {r.release} {r.version} {r.machine}'

    @staticmethod
    @_auth
    def version() -> str:
        return sys.version

    @staticmethod
    def escape(text) -> str:
        return escape(str(text))

    @staticmethod
    def html_escape(text) -> str:
        return html_escape(str(text))

    @staticmethod
    def hitokoto():
        return hitokoto()


Bridge = _Bridge()
