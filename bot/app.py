import asyncio
import os

from typing import TYPE_CHECKING

from aiohttp import ClientTimeout
from ptbcontrib.aiohttp_request import AiohttpRequest
from telegram.ext import Application, ApplicationBuilder, ContextTypes

from . import env
from .env import log

if TYPE_CHECKING:
    from telegram import Bot


async def _post_init(app: Application):
    if asyncio.iscoroutine(r := env.driver.post_init()):
        await r


async def _post_stop(_: Application):
    if asyncio.iscoroutine(r := env.driver.post_stop()):
        await r


async def handle_error(update, context: ContextTypes.DEFAULT_TYPE):
    env.driver.on_error(context.error)


def _build_app() -> Application:
    timeout = ClientTimeout(total=30)
    builder = (
        ApplicationBuilder()
        .token(os.environ['TELEGRAM_BOT_TOKEN'])
        .request(AiohttpRequest(connection_pool_size=5, client_timeout=timeout))
        .get_updates_request(AiohttpRequest(connection_pool_size=5, client_timeout=timeout))
        .concurrent_updates(True)
        .post_init(_post_init)
        .post_stop(_post_stop)
    )

    # Looks like http://127.0.0.1:8081/
    if url := os.environ.get('LOCAL_SERVER'):
        log.info('Using local server: %s', url)
        builder = builder.base_url(url + 'bot').base_file_url(url + 'file/bot').local_mode(True)

    app = builder.build()
    app.add_error_handler(handle_error)
    return app


app = _build_app()

bot: Bot = app.bot
create_task = app.create_task
