from typing import Iterable
from .context import trace

lex_errors = []


# Find all "code blocks" and "block lines" - lines that ends with `;`, which are treated
# like a `block_inner`.
def lex(text: str, *, block: bool = False) -> Iterable[tuple[bool, str | None]]:
    block_starts = []
    if block:
        # Pretend a `{` before the start.
        block_starts.append(-1)

    buf = []  # buffered fragments inside code blocks
    quote = None
    raw = False
    comment = False
    escape = False
    escaping_indices = []
    cursor = 0  # number of chars processed

    line_start = 0
    line_buf = []

    yield_kind = None
    yield_buf = []

    def yield_text(start: int, end: int):
        if start >= end:
            return

        # Remove escaping `\` before `{`.
        for i in escaping_indices:
            assert start < i < end
            yield_buf.append(text[start:i])
            start = i + 1
        escaping_indices.clear()

        yield_buf.append(text[start:end])

    for p, c in enumerate(text):
        if escape:
            escape = False
            continue

        if comment:
            if c != '\n':
                continue
            # Skip the comment.
            cursor = p
            comment = False

        if quote:
            # The escaping char `\` in string literals is rehandled by the
            # parser later. We don't remove it from the fragment.
            # But in raw literals, `\` is NOT special.
            if not raw and c == '\\':
                escape = True
            elif c == quote:
                quote = None
            continue

        # Raw literals starting with '`' are NOT enclosed and consumes all the
        # rest of the statement, i.e., until the next statement boundary.
        if raw:
            if c != ';' and c != '{' and c != '}':
                # Statement boundaries ';' and '{' and '}' terminate raw literals,
                # but they are escaped in (and only in) quotes.
                # Escaping char `\` is NOT special in raw literals.
                if c == '"' or c == "'":
                    quote = c
                continue
            raw = False
            chunk = text[cursor + 1 : p]
            trace('Raw literal: %s', chunk)
            buf.append('\'')
            buf.append(chunk.replace("'", r"\'"))
            buf.append('\'')
            cursor = p

        yield_kind = None
        yield_buf.clear()

        match c:
            case '{':
                if not block_starts:
                    yield_kind = False
                    yield_text(cursor, p)
                    cursor = p

                block_starts.append(p)
                trace('Block starts: %s', block_starts)

            case '}':
                if not block_starts or (block and len(block_starts) == 1):
                    # An plain '}' as text, when no block is open.
                    # But, when `block` is set, we cannot close the first (pretended) block here,
                    # or stuff afterwards would be "leaked" as text fragments and injected into
                    # our output.
                    # We leave it to cause a Lark parse error later.
                    continue
                trace('Block ends: %s', block_starts)
                block_starts.pop()
                if not block_starts:
                    chunk = text[cursor:p]
                    buf.append(chunk)
                    # Remove the block start '{'.
                    # We cannot save `p+1` directly when '{' is found, in case of
                    # unclosed blocks that are recovered as final texts later.
                    # The '{' must exist there, as we cannot reach here if `block` is set.
                    chunk = ''.join(buf)[1:].strip()
                    buf.clear()
                    if chunk:
                        trace('Lexed block: %r', chunk)
                        yield_kind = True
                        yield_buf.append(chunk)
                    cursor = p + 1

            case _:
                if block_starts:
                    match c:
                        case "'" | '"':
                            quote = c
                        case '#':
                            comment = True
                            # Flush up before the comment.
                            if chunk := text[cursor:p]:
                                buf.append(chunk)
                            cursor = p
                        case '`':
                            raw = True
                            # Flush up before the raw literal.
                            if chunk := text[cursor:p]:
                                buf.append(chunk)
                            cursor = p
                        case '\n':
                            line_start = None
                            if line_buf:
                                yield from line_buf
                                line_buf.clear()

                # We only handles escaping '\' chars inside quotes (which is inside blocks)
                # or outside any blocks (which escapes '{' and '}').
                # The latter ones here must be removed from the output fragments.
                elif c == '\\':
                    escape = True
                    escaping_indices.append(p)
                elif c == '\n':
                    if line_start is not None:
                        line = text[line_start:p]
                        trace('Line: %r', line)

                        if (stem := line.rstrip()).endswith(';'):
                            # Naked block line inside inline text.
                            trace('Naked block line: %r\n  Dropped: %s', line, line_buf)
                            line_buf.clear()
                            escaping_indices.clear()

                            yield False, text[cursor:line_start]
                            cursor = p + 1
                            yield from lex(stem, block=True)

                            # We are actually consuming the line break here, but yielding a `\n`
                            # each time would result in too many empty lines for consecutive block lines.
                            # `None` hints the implicit line break after the naked block here.
                            yield False, None

                            line_start = p + 1
                            continue

                    if line_buf:
                        yield from line_buf
                        line_buf.clear()

                    yield_kind = False
                    yield_text(cursor, p + 1)
                    cursor = line_start = p + 1

        if yield_kind is not None:
            if any('\n' in part for part in yield_buf):
                for fragment in yield_buf:
                    yield yield_kind, fragment
            else:
                for fragment in yield_buf:
                    trace('Preserving inline fragment: %s %r', yield_kind, fragment)
                    line_buf.append((yield_kind, fragment))

    # Flush final line buffer, pretending a EOF.
    if not block and line_start is not None:
        line = text[line_start:]
        trace('Final Line: %r', line)

        if (stem := line.rstrip()).endswith(';'):
            trace('Final naked block line: %r\n  Dropped: %s', line, line_buf)
            yield False, text[cursor:line_start]
            yield from lex(stem, block=True)
            return

    if line_buf:
        yield from line_buf

    # Pretend a final `}` if `block` is set.
    if block:
        assert block_starts
        if len(block_starts) > 1:
            # There are NO text fragments or unclosed blocks in `block` mode, as the
            # entire input is treated as a single block, where we are only expected
            # to preprocess comments and raw literals with the `buf`.
            # Trying to recover would leak text fragments inside the block, so we
            # also leave it to cause a Lark parse error later.
            msg = f'Unbalanced line block starting at position {block_starts} {cursor}'
            lex_errors.append(msg)
            trace('%s %s', msg, text)

        chunk = text[cursor:]
        buf.append(chunk)
        if chunk := ''.join(buf).strip():
            yield True, chunk
        return

    if block_starts:
        # Unclosed block! Reparse the inner of the first unclosed block to recover.
        msg = f'Unclosed block starting at position {block_starts} {cursor}'
        lex_errors.append(msg)
        trace('%s', msg)

        p = block_starts[0]
        if p >= 0:
            yield False, text[p]

        rest = text[p + 1 :]
        trace('Recovering rest: %s', rest)
        yield from lex(rest)
        return

    # `buf` must be empty, since `block_starts` is empty.
    assert not buf

    c = cursor
    for i in escaping_indices:
        assert i > c
        yield False, text[c:i]
        c = i + 1

    if chunk := text[c:]:
        yield False, chunk
