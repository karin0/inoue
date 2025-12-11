import re
import string
from collections import UserDict
from typing import Any, Callable, Iterable

from simpleeval import simple_eval

from db import db
from util import log

# Allowed types for context values, as allowed by `simpleeval` by default (except None).
type Value = str | int | float | bool


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


reg = re.compile(r'{(.+?)}|^([^\r\n]+?);$', re.DOTALL | re.MULTILINE)
all_punct = frozenset(string.punctuation.replace('-', '').replace('_', ''))
MAX_DEPTH = 20


def to_str(v: Value) -> str:
    if isinstance(v, bool):
        return '1' if v else '0'
    return str(v)


class RenderContext:
    def __init__(
        self,
        overrides: dict[str, Value],
        *,
        this_doc: tuple[int | None, str] | None = None,
    ):
        self.ctx = OverriddenDict(overrides)
        self.this_doc = this_doc
        self.depth: int = 0
        self.errors: list[str] = []
        self.first_doc_id: int | None = None
        self.doc_name: str | None = None

    def get_ctx(self, key: str, expr: str, *, as_str: bool = False) -> Value:
        if (val := self.ctx.get(key)) is None:
            self.errors.append('undefined: ' + expr)
            return ''

        return to_str(val) if as_str else val

    # Evaluate expression to a value.
    # This should not have side effects on `ctx`.
    def evaluate(
        self, expr: str, *, in_directive: bool = True, as_str: bool = False
    ) -> Value:
        log.debug('evaluating: %s', expr)
        # Empty
        if not (expr := expr.strip()):
            return ''

        # Literal: {`raw}
        if expr[0] == '`':
            return expr[1:]

        if len(expr) >= 2:
            # Script: {"1 + 1"}
            if expr[0] == expr[-1] == '\"':
                expr = expr[1:-1].strip()
                try:
                    v = simple_eval(expr, names=self.ctx.clone())
                except Exception as e:
                    self.errors.append(f'evaluation: {expr}: {type(e).__name__}: {e}')
                    v = ''
                else:
                    log.debug('evaluated script: %s -> %s', expr, v)
                    # Ensure a `Value`.
                    if v is None:
                        return ''

                # Not necessarily str!
                return to_str(v) if as_str else v

            # Literal: {'raw'}
            if expr[0] == expr[-1] == '\'':
                return expr[1:-1]

        # Db doc get: {:name} or {*name}
        if expr[0] == ':' or expr[0] == '*':
            # Recursion is allowed up to a limit.
            if self.depth >= MAX_DEPTH:
                self.errors.append('max depth exceeded: ' + expr)
                return ''

            key = expr[1:].strip()

            # Allow self-reference in `handle_render` and `handle_render_doc`,
            # where the doc is not saved yet.
            # Note that `doc_id` can be None in `handle_render`, where the
            # `doc_name` is transient and only used for recursion.
            if row := self.doc_name == key and self.this_doc or db.get_doc(key):
                doc_id, doc = row
            else:
                self.errors.append('undefined: ' + expr)
                return ''

            if self.first_doc_id is None:
                self.first_doc_id = doc_id

            self.depth += 1
            r = self.render(doc)
            self.depth -= 1
            return r

        # Context equality: {key=value} (by string representation)
        if (p := expr.find('=', 1)) >= 0:
            key = expr[:p].strip()
            val = self.evaluate(expr[p + 1 :].strip(), as_str=True)
            now = self.get_ctx(key, expr, as_str=True)
            return '1' if now == val else '0'

        # Context get: {$key} or {key} (out of a condition directive)
        if expr[0] == '$' or not in_directive:
            key = expr.lstrip('$').strip()
            val = self.get_ctx(key, expr, as_str=as_str)
            return val

        # Literal: {raw}
        return expr

    def _do_assignments(
        self, cmd: str, p1: int, p2: int, set: Callable[[str, Value], Any]
    ):
        key1 = cmd[:p1].strip()
        q = (p := p2) - 1
        while p < len(cmd):
            if (c := cmd[p]) == '=':
                q = p
            elif c in all_punct:
                break
            p += 1
        val = self.evaluate(cmd[q + 1 :].strip())
        set(key1, val)
        if keys := cmd[p2:q].strip():
            for key in keys.split('='):
                set(key.strip(), val)

    def _repl(self, m: re.Match) -> str:
        cmd: str = m[1] or m[2]
        if not (cmd and (cmd := cmd.strip())):
            return m[0]

        # Comment
        if cmd[0] == '#':
            return ''

        ctx = self.ctx
        errors = self.errors

        # Conditional: {cond ? true-directive : false-directive} (false part optional)
        # Omit `?=` which is handled inside directives and cannot occur in conditions
        if in_directive := (
            (p := cmd.find('?', 1)) >= 0 and p + 1 < len(cmd) and cmd[p + 1] != '='
        ):
            # Not `as_str`, so `False` is still treated falsey.
            val = self.evaluate(cmd[:p].strip(), in_directive=False)
            test = val and val != '0'

            # This overrides doc expansion like `? :<doc>`, where we have to
            # use `*<doc>` instead.
            if (q := cmd.find(':', p + 1)) != -1:
                cmd = cmd[p + 1 : q].strip() if test else cmd[q + 1 :].strip()
            elif test:
                cmd = cmd[p + 1 :].strip()
            else:
                return ''

        ret = None

        def handle_cmd(cmd: str):
            nonlocal ret

            # Empty directive
            if not cmd:
                return

            # Literal: {`raw}
            if cmd[0] == '`':
                if ret is None:
                    ret = cmd[1:]
                else:
                    errors.append('redundant literal: ' + cmd)
                return

            # Db name set: {name:}
            if cmd[-1] == ':':
                if self.depth == 0:
                    if self.doc_name:
                        errors.append('doc name redefined: ' + cmd)
                    else:
                        self.doc_name = cmd[:-1].strip()
                return

            # Context flag override: {+key1} means {key1:=1}, {/key2} means {key2:=0}
            # Also starts an unary chain: {+key1/key2/+key3*doc}
            # Other unary operators like `$:*` don't start a chain, as they yields
            # immediate values instead of preparing context.
            if cmd[0] == '+' or cmd[0] == '/':
                val = '1' if cmd[0] == '+' else '0'
                for i in range(2, len(cmd)):
                    if cmd[i] in all_punct:
                        key = cmd[1:i].strip()
                        ctx.setdefault_override(key, val)
                        if rest := cmd[i:]:
                            return handle_cmd(rest)
                        break
                else:
                    key = cmd[1:].strip()
                    ctx.setdefault_override(key, val)
                return

            for p, c in enumerate(cmd):
                if c in all_punct:
                    break

            # p is the first punctuation position, or len(cmd)
            if 0 < p < len(cmd):
                match cmd[p : p + 2]:
                    # Context override: {key:=value}
                    # a. Can never be overridden (except by markup buttons)
                    # b. Does not interrupt the natural order of flags detected in markup
                    # {key1:=key2=...=value} affects all keys
                    case ':=':
                        return self._do_assignments(
                            cmd, p, p + 2, ctx.setdefault_override
                        )

                    # Context set if empty: {key?=value}
                    # {key1?=key2=...=value} sets all keys to value if empty
                    case '?=':
                        return self._do_assignments(cmd, p, p + 2, ctx.setdefault)

                match cmd[p]:
                    # Context set: {key=value}
                    # {key1=key2=...=value} sets all keys to value
                    case '=':
                        return self._do_assignments(cmd, p, p + 1, ctx.__setitem__)

                    # Context inplace replacement: {key|pat1/rep1|pat2/rep2|...}
                    case '|':
                        key = cmd[:p].strip()
                        val = ctx.get(key)
                        if val is None:
                            return errors.append('undefined: ' + cmd)

                        parts = cmd[p + 1 :].split('|')
                        if not parts:
                            return errors.append('invalid replacements: ' + cmd)

                        r = str(val)
                        for part in parts:
                            try:
                                pat, rep = part.split('/', 1)
                            except ValueError:
                                errors.append('invalid replacement: ' + part)
                            else:
                                r = r.replace(pat.strip(), rep.strip())

                        ctx[key] = r
                        return

                    # Context swap: {key1^key2}
                    case '^':
                        key1 = cmd[:p].strip()
                        key2 = cmd[p + 1 :].strip()
                        val1 = ctx.get(key1)
                        val2 = ctx.get(key2)

                        self.set_ctx(key1, val2)
                        self.set_ctx(key2, val1)
                        return

            # Only the first expression is evaluated. Subsequent ones are discarded.
            if ret is None:
                ret = self.evaluate(cmd, in_directive=in_directive, as_str=True)
            else:
                errors.append('redundant expression: ' + cmd)

        # Find semicolons, but not inside quotes. Very trivial parsing.
        q0 = q1 = 1
        last = 0
        for p, c in enumerate(cmd):
            match c:
                case '\'':
                    q0 ^= 1
                case '\"':
                    q1 ^= 1
                case ';':
                    if q0 and q1:
                        handle_cmd(cmd[last:p].strip())
                        last = p + 1
        handle_cmd(cmd[last:].strip())

        return ret or ''

    def set_ctx(self, k: str, v: Value | None):
        if v is None:
            self.ctx.pop(k, None)
        else:
            self.ctx[k] = v

    def render(self, text: str) -> str:
        return reg.sub(self._repl, text).strip()

    def get_flag(self, key: str, default: bool = False) -> bool:
        v = self.ctx.get(key, default)
        return v and v != '0'
