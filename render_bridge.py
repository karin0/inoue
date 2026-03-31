import os
import sys

import subprocess
from typing import Callable, Concatenate
from render_core import Box, Context, Value

from util import escape, html_escape, log
from motto import hitokoto


class Signature(Box):
    def __init__(self, token: int):
        super().__init__()
        self.token = token

    def __repr__(self) -> str:
        return f'<Signature: {self.token}>'


class _Bridge(Box):
    def __init__(self) -> None:
        super().__init__()
        self.funcs: dict[str, Callable[..., Value | None]] = {}

    def __repr__(self) -> str:
        return '<Bridge>'

    def public[**P](
        self,
        func: Callable[P, Value | None],
    ) -> Callable[P, Value | None]:
        name = func.__name__.strip('_')
        self.funcs[name] = func
        return func

    def __call__[**P](
        self,
        func: Callable[P, Value | None],
    ) -> Callable[Concatenate[Context, P], Value | None]:
        name = func.__name__.strip('_')

        def wrapper(ctx: Context, *args: P.args, **kwargs: P.kwargs) -> Value | None:
            # See `RenderContext.__init__`.
            if isinstance(signature := ctx.get('_trusted'), Signature):
                log.debug(
                    'Bridge: authorized %s for %s\n  ctx = %s',
                    signature.token,
                    name,
                    ctx,
                )
                return func(*args, **kwargs)
            log.warning(
                'Bridge: unauthorized access to %s\n  ctx = %s',
                name,
                ctx,
            )
            raise PermissionError('unauthorized')

        self.funcs[name] = wrapper
        return wrapper

    def __getattr__(self, name: str) -> Callable[..., Value | None]:
        if name in self.funcs:
            return self.funcs[name]
        raise AttributeError(f'bad bridge call: {name}')


Bridge = _Bridge()


@Bridge
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


@Bridge
def _eval(expr: str):
    return eval(expr)


@Bridge
def uname() -> str:
    r = os.uname()
    return f'{r.sysname} {r.nodename} {r.release} {r.version} {r.machine}'


@Bridge
def version() -> str:
    return sys.version


@Bridge.public
def _escape(text) -> str:
    return escape(str(text))


@Bridge.public
def _html_escape(text) -> str:
    return html_escape(str(text))


@Bridge.public
def _hitokoto():
    return hitokoto()
