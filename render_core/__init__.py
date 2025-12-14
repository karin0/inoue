import os
from contextlib import contextmanager
from typing import Callable, Sequence, Type, TypeVar, cast

from lark import Lark, Token, Tree
from lark.visitors import Interpreter
from lark.exceptions import LarkError

from db import db
from render_core.lex import lex
from util import log, notify, shorten

from .context import (
    is_tracing,
    trace,
    Value,
    to_str,
    OverriddenDict,
    ScopedContext,
)
from .lex import lex, lex_errors

MAX_DEPTH = 20
MAX_GAS = 1000

parser = Lark.open(os.path.join(os.path.dirname(__file__), 'dsl.lark'), parser='lalr')

T = TypeVar('T', bound=Tree | Token)


def narrow(x: Tree | Token, target_type: Type[T]) -> T:
    assert isinstance(x, target_type), x
    return cast(T, x)


class Abort(Exception):
    pass


class Exit(Exception):
    pass


class Engine(Interpreter):
    def __init__(
        self,
        overrides: dict[str, Value],
        *,
        doc_id: int | None = None,
    ):
        super().__init__()
        self.ctx = OverriddenDict(overrides)
        self._doc_id = doc_id
        self._doc_text = ''

        self.errors: list[str] = []
        self.doc_name: str | None = None
        self.first_doc_id: int | None = None

        self._output: list[Value] = []
        self._root_output: list[str] = []
        self._depth: int = 0
        self._branch_depth: int = 0
        self._dirty: bool = False

        self._scope = ScopedContext(
            self.ctx,
            self._error,
            {
                '__file__': self._get_doc_func,
                'print': self._print_func,
                'exit': self._exit_func,
            },
        )
        self._tree: Tree | None = None
        self._aborted: bool = False
        self._gas: int = 0

    def _consume_gas(self):
        if self._gas >= MAX_GAS:
            self._error('out of gas')
            raise Abort()
        self._gas += 1

    def _put(self, text: str):
        if text:
            if not (self._dirty or text.isspace()):
                self._dirty = True
            self._output.append(text)

    # Keep the original type to preserve types in code_block as expression when possible.
    def _put_val(self, val: Value):
        if is_tracing and val != '':
            trace('Putting value: %r', val)
        if isinstance(val, str):
            self._put(val)
        else:
            self._dirty = True
            self._output.append(val)

    # The output implementation is responsible for filtering out any empty strings,
    # like in `_put` or `_unary_chain`.
    def _gather_output(self, out: list[Value], *, as_str: bool = False) -> Value:
        match len(out):
            case 0:
                return ''
            case 1:
                # Keep the original type for single output as possible.
                val = out[0]
                return to_str(val) if as_str else val
            case _:
                return ''.join(to_str(x) for x in out)

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
                except Exit as e:
                    trace('Rendering exited at fragment: %s', shorten(fragment))
                    for val in e.args:
                        self._put(to_str(val))
                    continue
                except Abort:
                    with notify.suppress():
                        log.warning(
                            'Rendering aborted at fragment: %s', shorten(fragment)
                        )
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

        r = self._gather_output(self._output, as_str=True)
        assert isinstance(r, str)
        trace('Rendered result: %r, Errors: %r', r, self.errors)

        if strip:
            return r.strip()

        r = r.lstrip()
        if len(s := r.rstrip()) != len(r) and s and r.find('\n', len(s)) >= 0:
            return s + '\n'

        return s

    def render(self, text: str) -> str:
        self._doc_text = text
        # Internal document expansion does not trim spaces.
        r = self._render(text, strip=True)
        if is_tracing:
            trace('Final rendered result: %r', r)
            self.ctx.debug()
        return r

    def get_flag(self, key: str, default: bool = False) -> bool:
        val = self.ctx.get(key, default)
        return bool(val) and val != '0'

    def visit(self, tree: Tree):
        assert isinstance(tree, Tree)
        self._tree = tree

        if is_tracing:
            trace(
                '[%d %d] %s: %d %s',
                self._depth,
                self._gas,
                tree.data,
                len(tree.children),
                tree.children,
            )

        self._consume_gas()
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
        log.debug('Error: %s: %s', msg, ctx)
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
            assert self._depth >= 0
            if self._depth == 0 and self._root_output:
                old_output.extend(self._root_output)
                self._root_output.clear()
            self._output = old_output
            self._branch_depth = old_branch_depth
            self._dirty = old_dirty

    def _print_func(self, *args):
        out = self._root_output if self._depth > 0 else self._output
        for val in args:
            out.append(to_str(val))

    def _exit_func(self, *args):
        raise Exit(*args)

    def _block_inner(self, tree: Tree):
        stmts = tree.children.pop()
        for ch in tree.children:
            if not isinstance(ch, Tree):
                continue
            match ch.data:
                # Db doc name set: {name:}
                case 'doc_def':
                    if self._depth == 0:
                        key = self._iden(ch.children[0])
                        if self.doc_name:
                            self._error(f'doc name redefined: {self.doc_name} -> {key}')
                        else:
                            trace('Set doc name: %s', key)
                            self.doc_name = key
                # Db doc expand: {:doc} (same as {*doc} in unary expressions)
                case 'doc_ref':
                    iden = self._iden(ch.children[-1])
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
                self.visit(stmt)

            if (rest := tree.children[1]) is not None:
                rest = narrow(rest, Tree)
                assert rest.data == 'stmt_list'
                tree = rest
            else:
                break

    # Conditional branch: {<cond> ? true-directive : false-directive} (false part optional)
    def branch(self, tree: Tree):
        cond = narrow(tree.children[0], Tree)

        # Allow undefined vars as falsy in conditions.
        # Note that `$var ? 1 : 0` still emits errors when `var` is undefined.
        val = self._expr(cond, permissive=True, allow_undef=True)
        test = val and val != '0'

        # The associativity of `branch` is too greedy and consumes all statements
        # in a recursive `stmt_list`, like in `a ? b ; c; d` being parsed as
        # `a ? {b ; c ; d}`.
        # We keep only the first stmt here and execute the rest unconditionally.
        left = tree.children[1]
        right = tree.children[2]
        if tree.children[3] is None:
            if right is not None:
                right = narrow(right, Tree)
                assert right.data == 'stmt_list'
                right, rest = right.children
                visit = not test
            else:
                left = narrow(left, Tree)
                assert left.data == 'stmt_list'
                left, rest = left.children
                visit = test
        else:
            if (to := left if test else right) is not None:
                to = narrow(to, Tree)
                assert to.data == 'stmt_list'
                self._branch_depth += 1
                try:
                    self._stmt_list(to)
                finally:
                    self._branch_depth -= 1
            return

        if (to := left if test else right) is not None:
            to = narrow(to, Tree)
            self._branch_depth += 1
            try:
                if visit:
                    self.visit(to)
                else:
                    assert to.data == 'stmt_list'
                    self._stmt_list(to)
            finally:
                self._branch_depth -= 1

        if rest is not None:
            rest = narrow(rest, Tree)
            assert rest.data == 'stmt_list'
            self._stmt_list(rest)

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
                name = self._dyn_iden(name)

            self._scope.push(name)
            try:
                self._block_inner(inner)
            finally:
                self._scope.pop()
        else:
            self._block_inner(inner)

    def expr(self, tree: Tree):
        # Expression statement: {<expr>}
        # The value is directly appended to output.
        if isinstance(inner := tree.children[0], Tree):
            match inner.data:
                case 'assign_or_equal':
                    # Hijack the AOE "expression". Different semantics as statements.
                    return self._assign(inner)
                case 'unary_chain':
                    # Optimize: flatten the unary chain directly to output, without capturing.
                    return self._unary_chain(inner, put=self._put_val)
                case 'code_block':
                    # Optimize: flatten the nested block.
                    return self._code_block(inner)

        val = self._expr(tree, permissive=self._branch_depth == 0)
        self._put_val(val)

    def _evaluate(self, tree: Tree | Token | None, *, permissive: bool = False) -> str:
        if tree is None:
            return ''
        s = self._expr(narrow(tree, Tree), as_str=True, permissive=permissive)
        assert isinstance(s, str), s
        return s

    def _expr(
        self,
        tree: Tree,
        *,
        as_str: bool = False,
        permissive: bool = False,
        allow_undef: bool = False,
    ) -> Value:
        '''
        Naked literals are treated as context vars when `permissive` is True.
        so we write {some_var}, {show_var ? 1 : 0} and {show_var ? $some_var : literal}.
        '''
        if is_tracing:
            trace(
                '[%d] _expr: %s permissive=%s as_str=%s',
                self._gas,
                tree,
                permissive,
                as_str,
            )
        self._consume_gas()

        assert len(tree.children) == 1
        ch = narrow(tree.children[0], Tree)
        while ch.data == 'expr':
            # Flatten nested expressions: {( ( ... ) )}
            permissive = False
            assert len(ch.children) == 1
            ch = narrow(ch.children[0], Tree)

        match ch.data:
            case 'unary_chain':
                out = []

                def put(val: Value):
                    if val != '':
                        out.append(val)

                self._unary_chain(ch, put=put)
                return self._gather_output(out, as_str=as_str)

            case 'assign_or_equal':
                return self._equal(ch)

            # Equality check: {key == val} (by string representation!)
            # `a==b` means `$a == 'b'`, and `a==$b` means `$a == $b`.
            case 'compare':
                left = self._evaluate(ch.children[0], permissive=True)
                right = self._evaluate(ch.children[1])
                return '1' if left == right else '0'

            # Nested block: {{ ... }}
            case 'code_block':
                if self._depth >= MAX_DEPTH:
                    self._error('block stack overflow')
                    raise Abort()

                with self._push():
                    self._code_block(ch)
                    return self._gather_output(self._output, as_str=as_str)

            case 'branch':
                if self._depth >= MAX_DEPTH:
                    self._error('block stack overflow')
                    raise Abort()

                with self._push():
                    self.branch(ch)
                    return self._gather_output(self._output, as_str=as_str)

            # Python expression: {"1 + 1"}
            # This should be non-mutating, i.e. side-effect free.
            case 'dq_lit':
                expr = (
                    narrow(ch.children[0], Token)
                    .value[1:-1]
                    .strip()
                    .replace('\\"', '"')
                    .replace('\\\\', '\\')
                )
                try:
                    val = self._scope.eval(expr)
                except Exit as e:
                    trace(f'evaluate: %r', e)
                    raise
                except Exception as e:
                    self._error(f'evaluate: {expr!r}: {type(e).__name__}: {e}')
                    return ''

                # Not necessarily str!
                return to_str(val) if as_str else val

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
                key = narrow(ch.children[0], Token).value.strip()
                if permissive and self._check_iden(key):
                    if allow_undef:
                        _, val = self._scope.resolve_raw(key, as_str=as_str)
                        return '' if val is None else val
                    return self._scope.get(key, as_str=as_str)
                return key

            case _:
                raise ValueError(f'Bad expr: {tree.pretty()}')

    def _check_iden(self, key: str) -> bool:
        if not key:
            self._error('empty identifier')
            return False

        if any(c.isspace() for c in key):
            self._error('bad identifier: ' + repr(key))
            return False

        return True

    # Lark didn't distinguish VAR from LITERAL, so we have to strip manually.
    def _iden(self, tree: Tree | Token) -> str:
        key = narrow(tree, Token).value.strip()
        self._check_iden(key)
        return key

    def _dyn_iden(self, tree: Tree | Token) -> str:
        if isinstance(tree, Tree):
            # Paren expression as dynamic identifier:
            # {k=a; $(k)} means {$a}.
            assert tree.data == 'expr'
            key = self._expr(tree, permissive=True, as_str=True)
            assert isinstance(key, str), key
            key = key.strip()
        else:
            key = tree.value.strip()

        self._check_iden(key)
        return key

    def _unary_chain(self, tree: Tree, *, put: Callable[[Value], None]):
        assert tree.children, tree
        for ch in tree.children:
            put(self._unary(narrow(ch, Tree)))

    def _unary(self, tree: Tree, *, as_str: bool = False) -> Value:
        op = narrow(tree.children[0], Token).value
        name = tree.children[1]
        trace('Unary: %s %s', op, name)
        key = self._dyn_iden(name)

        match op:
            case '+':
                self.ctx.setdefault_override(key, '1')
                return ''

            case '-':
                self.ctx.setdefault_override(key, '0')
                return ''

            # Variable read: {$key}
            case '$':
                return self._scope.get(key, as_str=as_str)

            # Db doc expand: {*doc} (same as {:doc})
            case '*':
                return self._doc_ref(key)

            case _:
                raise ValueError(f'Bad deref op: {op}')

    def _get_doc(self, key: str) -> tuple[int | None, str]:
        # Allow self-reference in `handle_render` and `handle_render_doc`,
        # where the doc is not saved yet.
        # Note that `doc_id` can be None in `handle_render`, where the
        # `doc_name` is transient and only used for recursion.
        if self.doc_name == key:
            return self._doc_id, self._doc_text
        if row := db.get_doc(key):
            return row
        self._error('undefined doc: ' + key)
        return None, ''

    # Used in dq_lit evaluation as `__file__` function.
    def _get_doc_func(self, key=None) -> str:
        if isinstance(key, str):
            return self._get_doc(key)[1]

        if isinstance(key, int):
            if self.first_doc_id is None:
                self._error('unknown doc')
                return ''
            if (doc := db.get_doc_by_id(self.first_doc_id)) is None:
                self._error(f'missing doc: {self.first_doc_id}')
                return ''
            return doc

        # Self-mapping when no argument is given.
        return self._doc_text

    def _doc_ref(self, key: str) -> str:
        doc_id, doc = self._get_doc(key)

        if self.first_doc_id is None:
            self.first_doc_id = doc_id

        # Recursion is allowed up to a limit.
        if self._depth >= MAX_DEPTH:
            self._error('stack overflow')
            raise Abort()

        with self._push():
            trace('Rendering doc: %s %s', key, doc_id)
            return self._render(doc)

    # Due to LALR limitations, AOE chains have very different semantics:
    # - In expression statements, they set context vars and return ''.
    # - Otherwise, they check whether all the assigned vars equal the final value, and return '1' or '0'.
    # But in both cases, the ASSIGN_OP besides the first one must be '='.
    def _resolve_aoe_chain(self, tree: Tree) -> tuple[Sequence[str], str, Tree | None]:
        keys = [self._dyn_iden(tree.children[0])]
        op = narrow(tree.children[1], Token).value
        val = None

        while rest := tree.children[2]:
            rest = narrow(rest, Tree)
            assert rest.data == 'expr'
            tree = rest.children[0]
            if not (isinstance(tree, Tree) and tree.data == 'assign_or_equal'):
                val = rest
                break
            keys.append(self._dyn_iden(tree.children[0]))
            if narrow(tree.children[1], Token).value != '=':
                self._error('bad assign op in chain: ' + op)

        return keys, op, val

    # Do real assignments here.
    def _assign(self, tree: Tree):
        keys, op, expr = self._resolve_aoe_chain(tree)
        trace('Assign: keys=%s op=%r expr=%s', keys, op, expr)

        match op:
            case '=':
                # Context set: {key=value}
                # {key1=key2=...=value} sets all keys to value
                f = self.ctx.__setitem__

            case ':=':
                # Context override: {key:=value}
                # a. Can never be overridden (except by markup buttons)
                # b. Does not interrupt the natural order of flags detected in markup
                # {key1:=key2=...=value} affects all keys
                f = self.ctx.setdefault_override

            case '?=':
                # Context set if empty (undefined or falsy): {key?=value}
                # {key1?=key2=...=value} assigns the value to the empty keys only
                # Note that the evaluation is deferred (conditional) in this case,
                # like a `OnceCell` initialization.
                val = '' if expr is None else None
                for key in keys:
                    key = self._scope.current_key(key)
                    old = self.ctx.touch(key)
                    if not old or old == '0':
                        if val is None:
                            val = self._expr(expr)  # type: ignore[assignment]
                        self.ctx[key] = val
                    else:
                        trace('"?=": Skip for non-empty key: %s = %r', key, old)
                if val is None:
                    trace('"?=": Skipped evaluation for %s', keys)
                return

            case _:
                raise ValueError(f'Bad assign op: {tree.pretty()}')

        val = self._expr(expr) if expr else ''
        for key in keys:
            # Cannot assign to outer scopes. No `global` or `nonlocal`.
            # However, outer vars can still be "mutated" via `swap` or `repl`.
            self._scope.set(key, val, f)

    # Equality test: {key1=key2=...=<expr>}, but not as expression statements.
    def _equal(self, tree: Tree) -> str:
        keys, op, expr = self._resolve_aoe_chain(tree)
        trace('Equal: keys=%s op=%r expr=%s', keys, op, expr)
        if op != '=':
            self._error('bad assign op in equality test: ' + op)

        final_val = self._expr(expr, as_str=True) if expr else ''
        for key in keys:
            val = self._scope.get(key, as_str=True)
            if val != final_val:
                return '0'

        return '1'

    # Context swap: {key1^key2}
    # To be "atomic", This allows modifying vars in outer scopes.
    def swap(self, tree: Tree):
        key1 = self._dyn_iden(tree.children[0])
        key2 = self._dyn_iden(tree.children[1])
        key1, val1 = self._scope.resolve_raw(key1)
        key2, val2 = self._scope.resolve_raw(key2)
        self._scope.set_or_del_raw(key1, val2)
        self._scope.set_or_del_raw(key2, val1)

    # Context inplace replacement: {key|pat1/rep1|pat2/rep2|...}
    def repl(self, tree: Tree):
        key = self._iden(tree.children[0])
        key, val = self._scope.resolve_raw(key, '', as_str=True)
        assert isinstance(val, str), val
        for pair in tree.children[1:]:
            pat, sub = pair.children
            pat = self._evaluate(pat)
            sub = self._evaluate(sub)

            # Write back each time.
            self.ctx[key] = val = val.replace(pat, sub)
