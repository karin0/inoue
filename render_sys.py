# For `eval`.
import os
import sys

import subprocess
from typing import Callable, Concatenate
from render_core import Box, Context

from util import escape, html_escape, log


class _Signature(Box):
    def __repr__(self) -> str:
        return '<Signature>'


Signature = _Signature()


class _Syscall(Box):
    def __repr__(self) -> str:
        return '<Syscall>'

    # Underscore to avoid being injected into rendering engine.
    @staticmethod
    def _auth[**P, T](
        func: Callable[P, T],
    ) -> Callable[Concatenate[Context, P], T]:

        # Not using `wraps()` to keep opaque.
        def wrapper(ctx: Context, *args: P.args, **kwargs: P.kwargs) -> T:
            # See `RenderContext.__init__`.
            if ctx.get('_trusted') is Signature and isinstance(
                doc_id := ctx.get('_doc_id'), int
            ):
                log.debug(
                    '_Syscall: authorized doc %d for %s\n  ctx = %s',
                    doc_id,
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
