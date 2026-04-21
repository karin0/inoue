import sys
import asyncio
from typing import Literal, overload

from util import log, is_debug


@overload
async def run_ffmpeg(
    *args: str,
    desc: str = '',
    prog: str = 'ffmpeg',
    capture: Literal[False] = False,
) -> None: ...


@overload
async def run_ffmpeg(
    *args: str,
    desc: str = '',
    prog: str = 'ffmpeg',
    capture: Literal[True] = True,
) -> bytes: ...


async def run_ffmpeg(
    *args: str, desc: str = '', prog: str = 'ffmpeg', capture: bool = False
) -> bytes | None:
    proc = await asyncio.create_subprocess_exec(
        prog,
        '-hide_banner',
        *(() if is_debug else ('-v', 'warning')),
        *args,
        stdout=asyncio.subprocess.PIPE if capture else None,
        stderr=None if is_debug else asyncio.subprocess.PIPE,
    )

    out, err = await proc.communicate()

    if err:
        sys.stderr.buffer.write(err)
        sys.stderr.buffer.flush()

    ret = proc.returncode
    if not desc:
        desc = prog

    if capture:
        log.info('%s finished with %s, output %s bytes', desc, ret, len(out))
    else:
        log.info('%s finished with %s', desc, ret)

    if ret:
        if err and (text := err.decode(errors='replace').strip()):
            msg = text[text.rfind('\n') + 1 :]
        else:
            msg = None
        raise RuntimeError(f'{desc} failed: {msg}')

    return out
