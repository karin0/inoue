import os
from typing import Callable, Awaitable

from telegram import Bot
from telegram.ext import Application, ApplicationBuilder, ContextTypes

from .env import log, db

_post_init_hooks: list[Callable[[], Awaitable]] = []
_error_hooks: list[Callable[[Exception | None], Awaitable]] = []

post_init = _post_init_hooks.append
on_error = _error_hooks.append


async def _post_init(app: Application) -> None:
    global _post_init_hooks

    db.connect()
    for hook in _post_init_hooks:
        await hook()
    _post_init_hooks.clear()
    del _post_init_hooks


async def _post_stop(_: Application) -> None:
    db.close()


async def handle_error(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    e = context.error
    for hook in _error_hooks:
        await hook(e)


def _build_app() -> Application:
    builder = (
        ApplicationBuilder()
        .token(os.environ['TELEGRAM_BOT_TOKEN'])
        .concurrent_updates(True)
        .post_init(_post_init)
        .post_stop(_post_stop)
        .read_timeout(30)
        .write_timeout(30)
        .media_write_timeout(60)
    )

    # Looks like http://127.0.0.1:8081/
    if url := os.environ.get('LOCAL_SERVER'):
        log.info('Using local server: %s', url)
        builder = (
            builder.base_url(url + 'bot')
            .base_file_url(url + 'file/bot')
            .local_mode(True)
        )

    app = builder.build()
    app.add_error_handler(handle_error)
    return app


app = _build_app()

bot: Bot = app.bot
create_task = app.create_task
