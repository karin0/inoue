import os
from contextlib import contextmanager
from typing import Callable, Sequence, Type, TypeVar, cast, override

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
    try_to_str,
    OverriddenDict,
    ScopedContext,
    ContextCallbacks,
)
from .lex import lex, lex_errors

MAX_DEPTH = 20
MAX_GAS = 2000

parser = Lark.open(os.path.join(os.path.dirname(__file__), 'dsl.lark'), parser='lalr')

T = TypeVar('T', bound=Tree | Token)


def narrow(x: Tree | Token, target_type: Type[T]) -> T:
    assert isinstance(x, target_type), x
    return cast(T, x)


class Abort(Exception):
    pass


class Exit(Abort):
    pass


class Engine(Interpreter, ContextCallbacks):
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
        self._doc_scope: str = ''

        self._tree: Tree | None = None
        self._gas: int = 0

        self._scope = ScopedContext(
            self.ctx,
            self,
            {
                '__file__': self._get_doc_func,
                'print': self._print_func,
                'exit': self._exit_func,
            },
        )

    @override
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
    def _gather_output(
        self, out: list[Value], *, as_str: bool = False, trim: bool = False
    ) -> Value:
        match len(out):
            case 0:
                return ''
            case 1:
                # Keep the original type for single output as possible.
                val = out[0]
                if as_str:
                    val = to_str(val)
                    return self._trim_output(val) if trim else val
                elif trim and isinstance(val, str):
                    return self._trim_output(val)
                else:
                    return val
            case _:
                val = ''.join(to_str(x) for x in out)
                return self._trim_output(val) if trim else val

    def _trim_output(self, s: str) -> str:
        r = s.lstrip()
        s = r.rstrip()
        if len(s) != len(r) and s and r.find('\n', len(s)) >= 0:
            return s + '\n'
        return s

    def _render(self, text: str):
        for is_block, fragment in lex(text):
            if lex_errors:
                self.errors.extend(lex_errors)
                lex_errors.clear()

            if is_block:
                assert fragment is not None
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
                    # "exit()" called: stop rendering, but only for *this fragment*.
                    # Fragments afterwards and outer docs will continue rendering.
                    trace('Rendering exited at fragment: %s', shorten(fragment))
                    continue
                except Abort:
                    # More serious "exit": stop the entire rendering, skipping all fragments,
                    # this and all outer docs.
                    with notify.suppress():
                        log.warning(
                            'Rendering aborted at fragment: %s', shorten(fragment)
                        )
                    raise
            else:
                trace('Appending text fragment: %r', fragment)
                if fragment is None:
                    # Merge consecutive line breaks from naked block lines without outputs.
                    if self._dirty:
                        self._output.append('\n')
                        self._dirty = False
                else:
                    self._put(fragment)

        trace('Rendered result: %r', self._output)

    def render(self, text: str) -> str:
        self._doc_text = text
        # Internal document expansion does not trim spaces.
        try:
            self._render(text)
        except Abort:
            trace('Rendering aborted.')
        r = self._gather_output(self._output, as_str=True)
        if is_tracing:
            trace('Final rendered result: %r', r)
            self.ctx.debug()
        return r.strip()  # type: ignore

    def get_flag(self, key: str, default: bool = False) -> bool:
        val = self.ctx.get(key, default)
        return bool(val) and val != '0'

    def debug_node(self, node: Tree | Token | None, depth: int = 0) -> str:
        if node is None:
            return '/'
        if isinstance(node, Token):
            return f'{node.type}({node.value})'
        assert isinstance(node, Tree)
        nr = len(node.children)
        if nr > 1:
            if depth > 2:
                return f'{node.data}[{nr}]'
            nr = f'[{nr}]'
        else:
            nr = ''
        chs = ', '.join(self.debug_node(ch, depth + 1) for ch in node.children)
        return f'{node.data}{nr}({chs})'

    @override
    def visit(self, tree: Tree):
        assert isinstance(tree, Tree)
        self._tree = tree

        if is_tracing:
            trace('[%d %d] %s', self._depth, self._gas, self.debug_node(tree))

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

    @override
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
        old_doc_scope = self._doc_scope
        self._depth += 1
        self._output = []
        self._branch_depth = 0
        self._dirty = False
        self._doc_scope = self._scope.current()
        err = False
        try:
            yield
        except Exception:
            err = True
            raise
        finally:
            self._depth -= 1
            trace('Pop: %d %s %s', self._depth, self._output, old_output)
            assert self._depth >= 0
            if self._depth == 0 and self._root_output:
                old_output.extend(self._root_output)
                self._root_output.clear()
            if err:
                old_output.extend(self._output)
            self._output = old_output
            self._branch_depth = old_branch_depth
            self._dirty = old_dirty
            self._doc_scope = old_doc_scope

    def _print_func(self, *args):
        out = self._root_output if self._depth > 0 else self._output
        first = True
        for val in args:
            if first:
                first = False
            else:
                out.append(' ')
            out.append(try_to_str(val))
        out.append('\n')

    def _exit_func(self):
        raise Exit()

    def _block_inner(self, tree: Tree):
        stmts = tree.children.pop()

        if (op := tree.children[0]) is not None:
            name = tree.children[1]
            assert narrow(op, Token).type == 'SCOPE_OP'
            if name is None:
                name = ''
            else:
                name = self._dyn_iden(name)

            self._scope.push(name)
            try:
                self.__block_inner(tree.children[2:], stmts)
            finally:
                self._scope.pop()
        else:
            self.__block_inner(tree.children[1:], stmts)

    def __block_inner(self, children: list[Tree], stmts: Tree | Token):
        for ch in children:
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
                    self._put_val(self._doc_ref(iden))
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
        # This only applies to the direct 'unary_chain', AOE and 'naked_lit' as
        # the condition expression.
        val = self._expr(cond, permissive=True, allow_undef=True)

        test = val and val != '0'

        # The associativity of `branch` is too greedy and consumes all statements
        # in a recursive `stmt_list`, like `a ? b ; c; d` being parsed as
        # `a ? {b ; c ; d}` by Lark.
        # We keep only the first stmt here and execute the rest unconditionally
        # to work around and make it more intuitive, behave like a "real" ternary
        # operator that accepts single expressions, i.e. `a ? b : c; d;` means
        # `(a ? b : c); d;`.
        # You can still explicitly specify the captured parts with braces or a
        # final marker, like in `a ? b : {c ; d}` or `a ? b : c ; d !`.
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

            if inner.children[0] is not None:
                self._error('double scope')
                inner.children[0] = inner.children[1] = None

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
            feats = (
                permissive and 'permissive',
                as_str and 'as_str',
                allow_undef and 'allow_undef',
            )
            if feats:
                feats = ' [' + ', '.join(f for f in feats if f) + ']'
            else:
                feats = ''
            trace(
                '[%d] _expr: %s%s',
                self._gas,
                self.debug_node(tree.children[0]),
                feats,
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

                self._unary_chain(ch, put=put, allow_undef=allow_undef)
                return self._gather_output(out, as_str=as_str)

            case 'assign_or_equal':
                return self._equal(ch, allow_undef=allow_undef)

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
                except Abort as e:
                    # This includes `Exit`.
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
                    return self._scope.get(key, as_str=as_str, allow_undef=allow_undef)
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

    def _unary_chain(
        self, tree: Tree, *, put: Callable[[Value], None], allow_undef: bool = False
    ):
        assert tree.children, tree
        for ch in tree.children:
            val = self._unary(narrow(ch, Tree), allow_undef=allow_undef)
            trace('Unary value: %r', val)
            put(val)

    def _get_by_raw_key(
        self, key: str, *, as_str: bool = False, allow_undef: bool = False
    ) -> Value:
        val = self.ctx.get(key)
        if val is None:
            if not allow_undef:
                self._error('undefined: ' + key)
            return ''
        return to_str(val) if as_str else val

    def _unary(
        self, tree: Tree, *, as_str: bool = False, allow_undef: bool = False
    ) -> Value:
        op = narrow(tree.children[0], Token).value
        name = tree.children[1]
        trace('Unary: %s %s', op, name)

        # `allow_undef` does affect nested expressions in dynamic identifiers.
        key = self._dyn_iden(name)

        match op:
            # Flag set: {+name} or {-name}
            # This means {name:=1} or {name:=0}.
            case '+':
                self.ctx.setdefault_override(key, '1')
                return ''

            case '-':
                self.ctx.setdefault_override(key, '0')
                return ''

            # Variable read: {$name}
            # Resolve from the current or the nearest outer scopes with the name defined.
            case '$':
                return self._scope.get(key, as_str=as_str, allow_undef=allow_undef)

            # Db doc expand: {*doc} (same as {:doc})
            case '*':
                return self._doc_ref(key)

            # Variable get in "root" scope: {::name}
            # Always resolve in the scope where the current doc (not necessarily
            # the first doc in the `_render` chain) begins.
            # Also, this returns '' if undefined.
            # This is useful for "recursive calls" to retrieve "arguments" passed
            # from outer docs.
            case '::':
                key = self._doc_scope + key
                return self._get_by_raw_key(key, as_str=as_str, allow_undef=True)

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

    def _doc_ref(self, key: str) -> Value:
        doc_id, doc = self._get_doc(key)

        if self.first_doc_id is None:
            self.first_doc_id = doc_id

        # Recursion is allowed up to a limit.
        if self._depth >= MAX_DEPTH:
            self._error('stack overflow')
            raise Abort()

        with self._push():
            trace('Rendering doc: %s %s', key, doc_id)
            self._render(doc)
            return self._gather_output(self._output, trim=True)

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
                # Context set if undefined or empty: {name?=value}
                # {name1?=name2=...=value} assigns the value to the undefined/empty
                # names only. Only '' (empty string) is considered empty.
                #
                # Mote that the evaluation is short-circuit (conditional) here,
                # like a `OnceCell` initialization.
                #
                # Also, be careful that the original value can be resolved from outer
                # scopes, in which case a key is still created in the current scope
                # to hold the outer value, even if the provided expression is not
                # evaluated!
                #
                # This is to ensure consistent behavior with other assignment ops
                # to ensure the var exists in the current scope after the assignment.
                val = '' if expr is None else None
                for key in keys:
                    raw_key, old = self._scope.resolve_raw(key)
                    key = self._scope.current_key(key)
                    if old is None or old == '':
                        if val is None:
                            val = self._expr(expr)  # type: ignore[assignment]
                        new = val
                    else:
                        new = old

                    trace(
                        'Assigning: "?=": key: %s = %r (%s was %r)',
                        key,
                        new,
                        raw_key,
                        old,
                    )

                    # Write back even if unchanged. This has two effects:
                    # 1. Ensures the var in the inner scope with the outer value.
                    # 2. Notifies the underlying `OverriddenDict` to touch the key.
                    #    See `OverriddenDict._setitem__`.
                    self.ctx[key] = new

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
    def _equal(self, tree: Tree, allow_undef: bool = False) -> str:
        keys, op, expr = self._resolve_aoe_chain(tree)
        trace('Equal: keys=%s op=%r expr=%s', keys, op, expr)
        if op != '=':
            self._error('bad assign op in equality test: ' + op)

        final_val = self._expr(expr, as_str=True) if expr else ''
        for key in keys:
            val = self._scope.get(key, as_str=True, allow_undef=allow_undef)
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
