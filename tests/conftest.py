import asyncio
import inspect
import os
import sys

import pytest

from . import fake_app

# `bot` and `inoue` read their configuration while being imported, so everything
# here happens before pytest imports a test module.
os.environ.update(
    ME='TestBot',
    USER_ID='111',
    CHAN_ID='222',
    GROUP_ID='333',
    MEDIA_STAGING_CHAT_ID='444',
    TELEGRAM_BOT_TOKEN='1:test',  # noqa: S106 -- a placeholder for inoue.log to redact
    DB_FILE=':memory:',
)
sys.modules['bot.app'] = fake_app
# A deployment may drop a `conf` module next to the package. Tests run against
# the defaults a clean checkout has.
sys.modules['conf'] = None  # pyright: ignore[reportArgumentType] -- None blocks the import


@pytest.fixture(scope='session', autouse=True)
def _runtime():
    from inoue.db import db
    from inoue.log import notify

    db.connect()
    with notify.suppress():
        yield
    db.close()


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    # Each coroutine test is the main task of its own loop, so `fakes.settle` can
    # wait on every other task. The anyio plugin keeps a runner task in the loop
    # that waits on the test itself.
    if inspect.iscoroutinefunction(func := pyfuncitem.obj):
        code = func.__code__
        names = code.co_varnames[: code.co_argcount]
        asyncio.run(func(**{name: pyfuncitem.funcargs[name] for name in names}))
        return True
    return None
