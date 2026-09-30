import mmap
import tempfile

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from inoue import rg
from inoue.rg import SECTION_GAP, SECTION_SEP_OFFSET, RGFile, RGMatch, RGQuery, Section, do_show

from .fakes import FakeResponder

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

H1 = b'>>> 2026-01-02 12:00:00 (1)\n'
H2 = b'>>> 2026-01-02 13:00:00 (2)\n'
H3 = b'>>> 2026-01-02 14:00:00 (3)\n'
LONG = H1 + b'a' * 20000 + b'\n' + H2 + b'b' * 20000 + b'\n' + H3 + b'c' * 20000


@contextmanager
def mapped(content: bytes) -> Iterator[mmap.mmap]:
    with tempfile.TemporaryFile() as fp:
        fp.write(content)
        fp.flush()
        with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            yield mm


def discover(mm: mmap.mmap, offset: int) -> Section:
    sect = Section.discover(mm, offset)
    assert sect is not None
    return sect


def test_short_section_is_shown_whole():
    content = b'header1\ncontent1\n' + H1 + b'message 1 body\n' + H2 + b'message 2 body\n'
    with mapped(content) as mm:
        sect = discover(mm, content.index(b'message 1 body'))
        assert sect.hit
        assert sect.start == content.index(H1)
        assert sect.end == content.index(H2)
        assert sect.decode(mm).startswith('> 2026-01-02 12:00:00 (1)')
        assert sect.next_offset == sect.start - SECTION_SEP_OFFSET - 1
        assert sect.prev_offset == sect.end + SECTION_GAP


def test_long_section_keeps_its_real_bounds():
    content = H1 + b'a' * 30000 + b'\n' + H2 + b'message 2 body\n'
    with mapped(content) as mm:
        sect = discover(mm, 20000)
        assert not sect.hit
        assert sect.start == 0
        assert sect.end == content.index(H2)
        assert sect.view_end <= sect.end
        assert sect.next_offset == 0
        assert discover(mm, sect.prev_offset).start == content.index(H2)


def test_navigation_walks_older_and_back():
    with mapped(LONG) as mm:
        sect3 = discover(mm, 55000)
        assert sect3.start == LONG.index(H3)
        assert sect3.end == len(LONG)

        sect2 = discover(mm, sect3.next_offset)
        assert sect2.start == LONG.index(H2)
        assert sect2.end == LONG.index(H3)

        sect1 = discover(mm, sect2.next_offset)
        assert sect1.start == 0
        assert sect1.end == LONG.index(H2)

        assert discover(mm, sect1.prev_offset).start == sect2.start


def test_navigation_walks_newer_and_back():
    with mapped(LONG) as mm:
        sect2 = discover(mm, 30000)
        assert sect2.start == LONG.index(H2)

        sect3 = discover(mm, sect2.prev_offset)
        assert sect3.start == LONG.index(H3)

        assert discover(mm, sect3.next_offset).start == sect2.start
        assert discover(mm, sect2.next_offset).start == 0


async def test_edge_sections_disable_their_outward_button(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    content = H1 + b'section 1 body\n' + H2 + b'section 2 body\n'
    (tmp_path / 'log.txt').write_bytes(content)
    matches = [
        RGMatch(
            text=text, line_number=line, absolute_offset=content.index(text.encode()), match='body'
        )
        for text, line in (('section 1 body', 2), ('section 2 body', 4))
    ]
    rs = FakeResponder()
    query = RGQuery(files=[RGFile(matches=matches, path='log.txt')], cwd=str(tmp_path), match_cnt=2)
    query.message = rs.handle
    monkeypatch.setattr(rg, 'QUERIES', [query])

    await do_show(rs, 0, 0, 0, None)
    assert len(rs.handle.edits) == 1
    prev_btn, _, next_btn = rs.handle.edits[-1][2].inline_keyboard[0]
    assert prev_btn.text == 'Prev'
    assert next_btn.text == ' '
    assert next_btn.callback_data == 'noop'

    await do_show(rs, 0, 0, 1, None)
    assert len(rs.handle.edits) == 2
    prev_btn, _, next_btn = rs.handle.edits[-1][2].inline_keyboard[0]
    assert prev_btn.text == ' '
    assert prev_btn.callback_data == 'noop'
    assert next_btn.text == 'Next'
