import os
import logging
from collections import UserDict
from contextlib import contextmanager
from typing import Iterable, Sequence, Type, TypeVar, cast

from simpleeval import simple_eval
from lark import Lark, Token, Tree
from lark.visitors import Interpreter
from lark.exceptions import LarkError

from db import db
from util import log, notify, shorten

# Allowed types for context values, as allowed by `simpleeval` by default (except None).
# Other types should have been disallowed in `simpleeval`.
type Value = str | int | float | bool | complex | bytes


class OverriddenDict(UserDict):
    def __init__(self, overrides: dict[str, Value]):
        super().__init__()
        self.overrides = overrides

    def __getitem__(self, key: str) -> Value:
        if (v := self.overrides.get(key)) is not None:
            return v
        return self.data[key]

    # For `get` method to work.
    def __contains__(self, key: str) -> bool:
        return key in self.overrides or key in self.data

    def setdefault(self, key: str, default: Value) -> Value:
        # For `?=` operator.
        #
        # We always set the default in the underlying dict, even if the value
        # is overridden, so the natural order of keys is preserved as defined by
        # documents and won't change after markup buttons are activated.
        #
        # For the same purpose, `__setitem__` (for `=` operator) is not overridden.
        r0 = self.data.setdefault(key, default)
        if (r := self.overrides.get(key)) is not None:
            return r
        return r0

    def setdefault_override(self, key: str, value: Value) -> Value:
        # For `:=` operator.
        #
        # This only affects the `overrides` dict, which has higher priority than
        # the underlying dict by their *values*, but are placed after those touched
        # by `=` and `?=` in the natural order of *keys*.
        #
        # There is no need to write back the overridden *value* either here or
        # in `__setitem__`, since `finalize()` already reflects the values from
        # `overrides`.
        return self.overrides.setdefault(key, value)

    def clone(self) -> 'OverriddenDict':
        new = OverriddenDict(self.overrides.copy())
        new.data = self.data.copy()
        return new

    def finalize(self) -> Iterable[tuple[str, Value]]:
        # Keys from the underlying dict are yielded first, in their natural order.
        r = self.data
        r.update(self.overrides)
        return r.items()


def to_str(v: Value) -> str:
    if isinstance(v, bool):
        return '1' if v else '0'
    return str(v)


MAX_DEPTH = 20
MAX_GAS = 1000

is_tracing = os.environ.get('TRACE') == '1' and log.isEnabledFor(logging.DEBUG)
trace = log.debug if is_tracing else lambda *_: None

parser = Lark.open('dsl.lark', parser='lalr')
lex_errors = []


# First stage: find all "block lines" - lines that ends with `;`, which are treated
# like a `block_inner`.
# For the other lines, we lex them again to find the real blocks `{ ... }` inside.
def lex(text: str) -> Iterable[tuple[bool, str | None]]:
    lines = []
    for line in text.splitlines(keepends=True):
        if (stem := line.rstrip()).endswith(';'):
            if lines:
                yield from _lex(''.join(lines))
                lines.clear()
            # Only `_lex` handles comments.
            yield from _lex(stem, block=True)

            if len(line) != len(stem):
                # We are actually consuming the line break here, but yielding a `\n`
                # each time would result in too many empty lines for consecutive block lines.
                # `None` hints the implicit line break after the naked block here.
                yield False, None
        else:
            lines.append(line)

    if lines:
        yield from _lex(''.join(lines))


# Thanks to the first stage, we no longer care about line endings.
def _lex(text: str, *, block: bool = False) -> Iterable[tuple[bool, str]]:
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

    for p, c in enumerate(text):
        if escape:
            escape = False
            continue

        if comment:
            if c != '\n' and c != ';' and c != '{' and c != '}':
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
            trace('Raw literal:\n%s', chunk)
            buf.append('\'')
            buf.append(chunk.replace("'", r"\'"))
            buf.append('\'')
            cursor = p

        match c:
            case '{':
                if not block_starts:
                    c = cursor
                    # Remove escaping `\` before `{`.
                    for i in escaping_indices:
                        assert c < i < p
                        yield False, text[c:i]
                        c = i + 1
                    escaping_indices.clear()
                    if c < p:
                        yield False, text[c:p]
                    cursor = p

                block_starts.append(p)

            case '}':
                if not block_starts or (block and len(block_starts) == 1):
                    # An plain '}' as text, when no block is open.
                    # But, when `block` is set, we cannot close the first (pretended) block here,
                    # or stuff afterwards would be "leaked" as text fragments and injected into
                    # our output.
                    # We leave it to cause a Lark parse error later.
                    continue
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
                        yield True, chunk
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

                # We only handles escaping '\' chars inside quotes (which is inside blocks)
                # or outside any blocks (which escapes '{' and '}').
                # The latter ones here must be removed from the output fragments.
                elif c == '\\':
                    escape = True
                    escaping_indices.append(p)

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
        yield from _lex(rest)
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


T = TypeVar('T', bound=Tree | Token)


def narrow(x: Tree | Token, target_type: Type[T]) -> T:
    assert isinstance(x, target_type), x
    return cast(T, x)


# LALR cannot distinguish VAR from LITERAL, so we have to strip manually.
def _iden(tree: Tree | Token) -> str:
    return narrow(tree, Token).value.strip()


class Abort(Exception):
    pass


class RenderInterpreter(Interpreter):
    def __init__(
        self,
        overrides: dict[str, Value],
        *,
        this_doc: tuple[int | None, str] | None = None,
    ):
        super().__init__()
        self.ctx = OverriddenDict(overrides)
        self.this_doc = this_doc

        self.errors: list[str] = []
        self.doc_name: str | None = None
        self.first_doc_id: int | None = None

        self._output: list[str] = []
        self._depth: int = 0
        self._branch_depth: int = 0
        self._dirty: bool = False

        self._tree: Tree | None = None
        self._scopes: list[str] = []
        self._aborted: bool = False
        self._gas: int = MAX_GAS

    def _put(self, text: str):
        if text:
            if not (self._dirty or text.isspace()):
                self._dirty = True
            self._output.append(text)

    def _render(self, text: str, strip: bool = False) -> str:
        for is_block, fragment in lex(text):
            if lex_errors:
                self.errors.extend(lex_errors)
                lex_errors.clear()

            if is_block:
                assert fragment is not None
                if self._aborted:
                    trace('Skipping aborted block: %r', fragment)
                    continue

                trace('Rendering block: %r', fragment)
                try:
                    tree = parser.parse(fragment)
                except LarkError as e:
                    self._error(f'parse: {fragment!r}: {type(e).__name__}: {e}')
                    continue

                if is_tracing:
                    trace('Parsed tree: %s', tree.pretty())

                try:
                    self.visit(tree)
                except Abort:
                    with notify.suppress():
                        log.error('Parsing aborted at fragment: %s', shorten(fragment))
                    self._aborted = True
                    continue
            else:
                trace('Appending text fragment: %r', fragment)
                if fragment is None:
                    # Merge consecutive line breaks from naked block lines without outputs.
                    if self._dirty:
                        self._output.append('\n')
                        self._dirty = False
                else:
                    self._put(fragment)

        r = ''.join(self._output)
        trace('Rendered result: %r, Errors: %r', r, self.errors)

        if strip:
            return r.strip()

        r = r.lstrip()
        if len(s := r.rstrip()) != len(r) and s and r.find('\n', len(s)) >= 0:
            return s + '\n'

        return s

    def render(self, text: str) -> str:
        if not self.this_doc:
            self.this_doc = (None, text)
        # Internal document expansion does not trim spaces.
        return self._render(text, strip=True)

    def get_flag(self, key: str, default: bool = False) -> bool:
        val = self.ctx.get(key, default)
        return bool(val) and val != '0'

    def visit(self, tree: Tree):
        if self._gas <= 0:
            self._error('out of gas')
            raise Abort()
        self._gas -= 1

        assert isinstance(tree, Tree)
        self._tree = tree

        if is_tracing:
            trace(
                '[%d %d] %s: %d %s',
                self._depth,
                MAX_GAS - self._gas,
                tree.data,
                len(tree.children),
                tree.children,
            )
        return super().visit(tree)

    # Override lark visitor to raise on unhandled nodes.
    def __getattr__(self, name: str):
        raise NotImplementedError(name)

    def start(self, tree: Tree):
        ch = narrow(tree.children[0], Tree)
        assert ch.data == 'block_inner'
        return self._block_inner(ch)

    def _tree_ctx(self) -> str:
        if self._tree is None:
            return 'unknown'
        return str(self._tree.pretty(indent_str='').replace('\n', ' '))

    def _error(self, msg: str):
        ctx = self._tree_ctx()
        trace('Error: %s: %s', msg, ctx)
        self.errors.append(f'{msg}: {ctx}')

        if len(self.errors) > 5:
            self.errors.append('too many errors, aborting')
            raise Abort()

    @contextmanager
    def _push(self):
        trace('Push: %d %s', self._depth, self._output)
        old_output = self._output
        old_branch_depth = self._branch_depth
        old_dirty = self._dirty
        self._depth += 1
        self._output = []
        self._branch_depth = 0
        self._dirty = False
        try:
            yield
        finally:
            self._depth -= 1
            trace('Pop: %d %s %s', self._depth, self._output, old_output)
            self._output = old_output
            self._branch_depth = old_branch_depth
            self._dirty = old_dirty

    def _block_inner(self, tree: Tree):
        stmts = tree.children.pop()
        for ch in tree.children:
            if not isinstance(ch, Tree):
                continue
            match ch.data:
                # Db doc name set: {name:}
                case 'doc_def':
                    if self._depth == 0:
                        key = _iden(ch.children[0])
                        if self.doc_name:
                            self._error(f'doc name redefined: {self.doc_name} -> {key}')
                        else:
                            trace('Set doc name: %s', key)
                            self.doc_name = key
                # Db doc expand: {:doc} (same as {*doc} in unary expressions)
                case 'doc_ref':
                    iden = _iden(ch.children[-1])
                    self._put(self._doc_ref(iden))
                case _:
                    raise ValueError(f'Bad block_inner child: {ch}')

        stmts = narrow(stmts, Tree)
        assert stmts.data == 'stmt_list', stmts
        self._stmt_list(stmts)

    def _stmt_list(self, tree: Tree):
        # Flatten right-recursion.
        while True:
            if (stmt := tree.children[0]) is not None:
                if (stmt := narrow(stmt, Tree)).data == 'stmt_list':
                    tree = stmt
                    continue
                self.visit(stmt)

            if len(tree.children) > 1 and (rest := tree.children[1]) is not None:
                rest = narrow(rest, Tree)
                assert rest.data == 'stmt_list'
                tree = rest
            else:
                break

    # Conditional branch: {<cond> ? true-directive : false-directive} (false part optional)
    def branch(self, tree: Tree):
        cond = narrow(tree.children[0], Tree)
        assert cond.data == 'expr'

        val = self._expr(cond, permissive=True)
        if val and val != '0':
            branch = tree.children[1]
        elif not (branch := tree.children[2]):
            return

        to = narrow(branch, Tree)
        assert to.data == 'stmt_list'
        self._branch_depth += 1
        self._stmt_list(to)
        self._branch_depth -= 1

    # Nested block: {{ ... }}
    # With scope: {@name {...}} (name optional, defaults to '_')
    def _code_block(self, tree: Tree):
        inner = narrow(tree.children[2], Tree)
        assert inner.data == 'block_inner'

        if (op := tree.children[0]) is not None:
            assert narrow(op, Token).type == 'SCOPE_OP'
            if (name := tree.children[1]) is None:
                name = ''
            else:
                name = _iden(name)

            if self._scopes:
                new_scope = self._scopes[-1] + name + '.'
            else:
                new_scope = name + '.'

            trace('Enter scope: %s -> %s', self._scopes, new_scope)
            self._scopes.append(new_scope)
            self._block_inner(inner)
            self._scopes.pop()
            trace('Exit scope: %s -> %s', new_scope, self._scopes)
        else:
            self._block_inner(inner)

    def expr(self, tree: Tree):
        # Expression statement: {<expr>}
        # The value is directly appended to output.
        if isinstance(inner := tree.children[0], Tree):
            if inner.data == 'code_block':
                # Flatten the nested block without capturing.
                return self._code_block(inner)
            elif inner.data == 'assign':
                # Assignment statement
                return self._assign_stmt(inner)

        val = self._expr(tree, permissive=self._branch_depth == 0, as_str=True)
        assert isinstance(val, str), val
        self._put(val)

    def _get_key_var(
        self, key: str, default: Value | None = '', *, as_str: bool = False
    ) -> tuple[str, Value | None]:
        if self._scopes:
            for scope in reversed(self._scopes):
                new_key = scope + key
                if (val := self.ctx.get(new_key)) is not None:
                    key = new_key
                    break
            else:
                if (val := self.ctx.get(key)) is None:
                    scope = self._scopes[-1]
                    if default is not None:
                        self._error(f'undefined: {key} @ {scope}')
                    return scope + key, default
        elif (val := self.ctx.get(key)) is None:
            if default is not None:
                self._error('undefined: ' + key)
            return key, default

        trace('Got var: %s = %r', key, val)
        return key, to_str(val) if as_str else val

    def _get_var(self, key: str, *, as_str: bool = False) -> Value:
        key, val = self._get_key_var(key, as_str=as_str)
        assert val is not None, key
        return val

    def _evaluate(self, tree: Tree | Token | None, *, permissive: bool = False) -> str:
        if tree is None:
            return ''
        s = self._expr(narrow(tree, Tree), as_str=True, permissive=permissive)
        assert isinstance(s, str), s
        return s

    def _expr(
        self, tree: Tree, *, as_str: bool = False, permissive: bool = False
    ) -> Value:
        '''
        Naked literals are treated as context vars when `permissive` is True.
        so we write {some_var}, {show_var ? 1 : 0} and {show_var ? $some_var : literal}.
        '''
        if is_tracing:
            trace(
                '[%d] _expr: %s permissive=%s as_str=%s',
                MAX_GAS - self._gas,
                tree,
                permissive,
                as_str,
            )

        if self._gas <= 0:
            self._error('out of gas')
            raise Abort()
        self._gas -= 1

        assert len(tree.children) == 1
        if (ch := tree.children[0]) is None:
            # Empty parens: {( )}
            return ''
        ch = narrow(ch, Tree)

        match ch.data:
            case 'unary_chain':
                return self._unary_chain(ch, as_str=as_str)

            case 'assign':
                return self._assign_expr(ch)

            # Equality check: {key == val} (by string representation!)
            # `a==b` means `$a == 'b'`, and `a==$b` means `$a == $b`.
            case 'compare':
                left = self._evaluate(ch.children[0], permissive=True)
                right = self._evaluate(ch.children[1])
                return '1' if left == right else '0'

            # Nested expression: {( ... )}
            case 'expr':
                return self._expr(ch, as_str=as_str)

            # Nested block: {{ ... }}
            case 'code_block':
                if self._depth >= MAX_DEPTH:
                    self._error('block stack overflow')
                    return ''

                with self._push():
                    self._code_block(ch)
                    return ''.join(self._output).strip()

            # Python expression: {"1 + 1"}
            case 'dq_lit':
                expr = (
                    narrow(ch.children[0], Token)
                    .value[1:-1]
                    .strip()
                    .replace('\\"', '"')
                    .replace('\\\\', '\\')
                )
                try:
                    v = simple_eval(expr, names=self.ctx.clone())
                except Exception as e:
                    self._error(f'evaluate: {expr!r}: {type(e).__name__}: {e}')
                    v = ''
                else:
                    trace('evaluated script: %s -> %s', expr, v)
                    # Ensure a `Value`.
                    if v is None:
                        return ''

                # Not necessarily str!
                return to_str(v) if as_str else v

            # Literal: {'single quoted'}
            case 'sq_lit':
                return (
                    narrow(ch.children[0], Token)
                    .value[1:-1]
                    .replace("\\'", "'")
                    .replace('\\\\', '\\')
                )

            # Ambiguous naked literal / var: {some_name}
            # Treated as var only if `permissive` is True and contains no whitespace.
            case 'naked_lit':
                key = _iden(ch.children[0])
                if permissive and not any(c.isspace() for c in key):
                    return self._get_var(key, as_str=as_str)
                return key

            case _:
                raise ValueError(f'Bad expr: {tree.pretty()}')

    def _unary_chain(self, tree: Tree, *, as_str: bool = False) -> Value:
        if len(tree.children) == 1:
            return self._unary(narrow(tree.children[0], Tree), as_str=as_str)

        out = []
        assert tree.children, tree
        for ch in tree.children:
            val = self._unary(narrow(ch, Tree), as_str=True)
            assert isinstance(val, str)
            trace('Unary chain part: %r', val)
            out.append(val)

        return ''.join(out)

    def _unary(self, tree: Tree, *, as_str: bool = False) -> Value:
        op = narrow(tree.children[0], Token).value
        key = _iden(tree.children[1])

        match op:
            case '+':
                self.ctx.setdefault_override(key, '1')
                return ''

            case '-':
                self.ctx.setdefault_override(key, '0')
                return ''

            # Variable read: {$key}
            case '$':
                return self._get_var(key, as_str=as_str)

            # Db doc expand: {*doc} (same as {:doc})
            case '*':
                return self._doc_ref(key)

            case _:
                raise ValueError(f'Bad deref op: {op}')

    def _doc_ref(self, key: str) -> str:
        # Allow self-reference in `handle_render` and `handle_render_doc`,
        # where the doc is not saved yet.
        # Note that `doc_id` can be None in `handle_render`, where the
        # `doc_name` is transient and only used for recursion.
        if row := self.doc_name == key and self.this_doc or db.get_doc(key):
            doc_id, doc = row
        else:
            self._error('undefined doc: ' + key)
            return ''

        if self.first_doc_id is None:
            self.first_doc_id = doc_id

        # Recursion is allowed up to a limit.
        if self._depth >= MAX_DEPTH:
            self._error('stack overflow')
            return ''

        with self._push():
            trace('Rendering doc: %s %s', key, doc_id)
            return self._render(doc)

    # Due to LALR limitations, assignment chains have very different semantics:
    # - In expression statements, they set context vars and return ''.
    # - Otherwise, they check whether all the assigned vars equal the final value, and return '1' or '0'.
    # But in both cases, the ASSIGN_OP besides the first one must be '='.
    def _resolve_assign_chain(
        self, tree: Tree
    ) -> tuple[Sequence[str], str, Tree | None]:
        keys = [_iden(tree.children[0])]
        op = narrow(tree.children[1], Token).value
        val = None

        while rest := tree.children[2]:
            rest = narrow(rest, Tree)
            assert rest.data == 'expr'
            tree = rest.children[0]
            if not (isinstance(tree, Tree) and tree.data == 'assign'):
                val = rest
                break
            keys.append(_iden(tree.children[0]))
            if narrow(tree.children[1], Token).value != '=':
                self._error('bad assign op in chain: ' + op)

        return keys, op, val

    # Do real assignments here.
    def _assign_stmt(self, tree: Tree):
        keys, op, expr = self._resolve_assign_chain(tree)
        trace('Assign stmt: keys=%s op=%r expr=%s', keys, op, expr)

        match op:
            case '=':
                # Context set: {key=value}
                # {key1=key2=...=value} sets all keys to value
                f = self.ctx.__setitem__

            case '?=':
                # Context set if empty: {key?=value}
                # {key1?=key2=...=value} sets all keys to value if empty
                f = self.ctx.setdefault

            case ':=':
                # Context override: {key:=value}
                # a. Can never be overridden (except by markup buttons)
                # b. Does not interrupt the natural order of flags detected in markup
                # {key1:=key2=...=value} affects all keys
                f = self.ctx.setdefault_override

            case _:
                raise ValueError(f'Bad assign op: {tree.pretty()}')

        val = self._expr(expr) if expr else ''
        for key in keys:
            if self._scopes:
                # Cannot assign to outer scopes. No `global` or `nonlocal`.
                key = self._scopes[-1] + key
            trace('Assigning: %s = %r', key, val)
            f(key, val)

    # Equality test: {key1=key2=...=<expr>}, but not as expression statements.
    def _assign_expr(self, tree: Tree) -> str:
        keys, op, expr = self._resolve_assign_chain(tree)
        trace('Assign expr: keys=%s op=%r expr=%s', keys, op, expr)
        if op != '=':
            self._error('bad assign op in equality test: ' + op)

        final_val = self._expr(expr, as_str=True) if expr else ''
        for key in keys:
            val = self._get_var(key, as_str=True)
            if val != final_val:
                return '0'

        return '1'

    def _set_var(self, key: str, val: Value | None):
        if val is None:
            self.ctx.pop(key, None)
        else:
            self.ctx[key] = val

    # Context swap: {key1^key2}
    def swap(self, tree: Tree):
        key1 = _iden(tree.children[0])
        key2 = _iden(tree.children[1])
        key1, val1 = self._get_key_var(key1, None)
        key2, val2 = self._get_key_var(key2, None)
        self._set_var(key1, val2)
        self._set_var(key2, val1)

    # Context inplace replacement: {key|pat1/rep1|pat2/rep2|...}
    def repl(self, tree: Tree):
        key = _iden(tree.children[0])
        key, val = self._get_key_var(key, as_str=True)
        assert isinstance(val, str)
        for pair in tree.children[1:]:
            pat, sub = pair.children
            pat = self._evaluate(pat)
            sub = self._evaluate(sub)

            # Write back each time.
            self.ctx[key] = val = val.replace(pat, sub)
