from unittest.mock import MagicMock

import pytest

from bot.inline_responder import InlineResponder
from inoue.ctx import use_context
from inoue.merge import handle_merge

from .fakes import HOST


def test_inline_message_has_no_incoming_message():
    rs = InlineResponder('abc')
    assert rs.get_message() is None
    assert rs.get_text() == ''
    assert rs.get_effective_arg() == ''
    assert rs.get_message_key() == 'abc'


def test_route_needing_a_message_refuses_an_inline_message():
    with (
        use_context(MagicMock(name='update'), None, HOST),
        pytest.raises(ValueError, match='needs a message'),
    ):
        handle_merge.route(InlineResponder('abc'))
