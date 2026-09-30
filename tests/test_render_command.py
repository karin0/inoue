from unittest.mock import MagicMock

import pytest

from inoue.ctx import use_context
from inoue.db import db
from inoue.render import handle_render

from .fakes import HOST, FakeResponder, settle


@pytest.fixture(autouse=True)
def _host():
    update = MagicMock(name='update', effective_user=None, effective_chat=None)
    update.effective_message = None
    with use_context(update, None, HOST):
        yield


async def test_source_is_stored_under_the_message_key():
    rs = FakeResponder('/render {1 + 1}')
    await handle_render(rs)
    await settle()
    assert db.get('r-#' + rs.get_message_key()) == '{1 + 1}'
    [(text, _, _)] = rs.replies
    assert text is not None
    assert text.startswith('```\n2```')


async def test_captured_render_keeps_the_capturing_source():
    rs = FakeResponder('/render outer')
    await handle_render(rs)
    await settle()

    rs.set_text('/render inner')
    with rs.capture() as buf:
        await handle_render(rs)
        await settle()
    [(text, _)] = buf
    assert text.startswith('```\ninner```')
    assert db.get('r-#' + rs.get_message_key()) == 'outer'
