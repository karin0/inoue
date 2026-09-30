from telegram.constants import KeyboardButtonStyle

from inoue.render import (
    CALLBACK_SIGNS,
    MEMORY_SIGN,
    PATH_SIGNS,
    encode_flags,
    is_doc_ref,
    is_safe_key,
    is_safe_mem,
    make_markup,
)
from inoue.render_ctx import BUTTON_KEY

from .fakes import make_data


def test_is_safe_key_rejects_callback_special_chars():
    assert is_safe_key('foo')
    assert is_safe_key('_foo_2')
    assert not is_safe_key('')
    assert not is_safe_key(None)
    for c in PATH_SIGNS + CALLBACK_SIGNS + MEMORY_SIGN:
        assert not is_safe_key(f'a{c}b'), c


def test_is_safe_mem_allows_callback_signs_but_not_paths():
    # A memory cell holds an encoded value, and path signs end it in callback data.
    assert is_safe_mem('i42')
    assert is_safe_mem(f's{CALLBACK_SIGNS}{MEMORY_SIGN}')
    for c in PATH_SIGNS:
        assert not is_safe_mem(f'i{c}'), c


def test_encode_flags_empty():
    assert encode_flags(None) == ''
    assert encode_flags({}) == ''


def test_encode_flags_keeps_insertion_order():
    assert encode_flags({'a': True, 'b': False, 'c': True}) == '+a-b+c'


def test_make_markup_returns_none_when_path_too_long():
    long_path = '`' + 'x' * 200
    markup, current = make_markup(long_path, make_data(), None, None, None)
    assert markup is None
    assert current is None


def test_make_markup_empty_when_no_inputs():
    markup, current = make_markup('`x', make_data(), None, None, None)
    assert markup is None
    assert current is None


def test_make_markup_renders_flags():
    markup, _ = make_markup('`x', make_data({'on': '1', 'off': '0'}), None, None, None)
    assert markup is not None
    labels = [btn.text for btn in markup.inline_keyboard[0]]
    assert 'x' in labels[0]
    # A label shows the current value, and the callback data carries the flipped one.
    assert any('on=1' in lbl for lbl in labels)
    assert any('off=0' in lbl for lbl in labels)


def test_make_markup_renders_explicit_buttons():
    ctx = make_data({'_env.btn.go': '1', '_env.btn.skip': '1', '_env.btn.disabled': '0'})
    markup, _ = make_markup('`x', ctx, None, None, None)
    assert markup is not None
    labels = {btn.text for row in markup.inline_keyboard for btn in row}
    assert 'go' in labels
    assert 'skip' in labels
    assert 'disabled' not in labels


def test_make_markup_uses_icon_label_for_button():
    ctx = make_data({'_env.btn.go': '1', '_env.icon.go': '▶'})
    markup, _ = make_markup('`x', ctx, None, None, None)
    assert markup is not None
    labels = {btn.text for row in markup.inline_keyboard for btn in row}
    assert '▶' in labels
    assert 'go' not in labels


def test_make_markup_external_button_first():
    ctx = make_data({'_env.btn.go': '1'})
    markup, _ = make_markup('`x', ctx, None, None, ('🛑3', '_cancel'))
    assert markup is not None
    row = markup.inline_keyboard[0]
    assert row[1].text == '🛑3'
    assert any(btn.text == 'go' for btn in row)


def test_make_markup_ignores_env_and_special_keys():
    ctx = make_data({'_env.atexit': '1', BUTTON_KEY: '_cancel', '_mem': 'X'})
    markup, _ = make_markup('`x', ctx, None, None, None)
    assert markup is None


def test_make_markup_chunks_into_rows_by_btns_per_row():
    # The path header and 7 buttons fill rows of 5.
    ctx = make_data({f'_env.btn.b{i}': '1' for i in range(7)})
    markup, _ = make_markup('`x', ctx, None, None, None)
    assert markup is not None
    rows = markup.inline_keyboard
    assert len(rows) == 2
    assert len(rows[0]) == 5
    assert len(rows[1]) == 3


def test_make_markup_respects_explicit_btns_per_row():
    ctx = make_data({**{f'_env.btn.b{i}': '1' for i in range(5)}, '_env.btns_per_row': '2'})
    markup, _ = make_markup('`x', ctx, None, None, None)
    assert markup is not None
    rows = markup.inline_keyboard
    assert [len(row) for row in rows] == [2, 2, 2]


def test_make_markup_invalid_btns_per_row_falls_back_to_default():
    ctx = make_data({**{f'_env.btn.b{i}': '1' for i in range(7)}, '_env.btns_per_row': 'x'})
    markup, _ = make_markup('`x', ctx, None, None, None)
    assert markup is not None
    assert len(markup.inline_keyboard[0]) == 5


def test_make_markup_with_state_marks_flipped_flags():
    state = ({'on': True}, '`x')
    markup, current = make_markup('`x', make_data({'on': '1'}), state, None, None)
    assert markup is not None
    assert current is None
    btn = next(b for row in markup.inline_keyboard for b in row if 'on' in b.text)
    assert ':=' in btn.text
    assert btn.style == KeyboardButtonStyle.SUCCESS


def test_make_markup_returns_current_when_state_diverges():
    state = ({}, '`other')
    markup, current = make_markup('`x', make_data({'flag': '1'}), state, None, None)
    assert markup is not None
    assert current == '`other'
    assert any(b.text == '🔄' for row in markup.inline_keyboard for b in row)


def test_is_doc_ref_no_match_returns_none():
    assert is_doc_ref('plain text') is None
    assert is_doc_ref('foo;bar') is None
