"""Exact circuits for Q-164M (architecture "Q"): the model *writes* a calculation/tool span; a fixed
program fills in the answer in the same token stream. Same mechanism as Q-Omni's circuits.py (explicit
trigger token + external deterministic executor, not implicit/schema-based tool calling -- see
q-agent/RESEARCH.md for why that split matters at this parameter count), renumbered because Q-164M has no
image/audio special tokens to reserve space for, plus two new tool kinds (`dur:`, `filter:`) as the first
step of generalizing beyond arithmetic per q-agent/PLAN.md Path C.
Span markup:  <CALC> expr <EQ> result <ECALC>
expr kinds: arithmetic "347*86" | "date:2026-12-20+45" | "wd:2027-03-05" | "dur:2026-01-01,2026-03-15" |
  "conv:10 km>mi" | "cmp:5.3,5.29" | "sort:5,2,9" | "filter:>5:1,6,3,8,2" | "count:banana" |
  "mean/max/min/sum:1,2,3" | "gcd:12,18" | "lcm:4,6" | "prime:97" | "rev:hello" | "fact:5" | "sqrt:2" |
  "pchg:100,120" | "words:a short sentence"
Inside a span every character is its own token (so digits are single tokens), outside spans text tokenises normally."""
import ast, datetime as dt, operator as op

USER, MODEL, EOT, THINK, ETHINK, CALC, EQ, ECALC, NEED, QUOTE = range(32768, 32778)
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg}
_UNITS = {("km", "mi"): 0.621371, ("mi", "km"): 1.609344, ("kg", "lb"): 2.204623, ("lb", "kg"): 0.453592, ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
          ("l", "gal"): 0.264172, ("gal", "l"): 3.78541, ("cm", "in"): 0.393701, ("in", "cm"): 2.54}
_CMP_OPS = {">": op.gt, "<": op.lt, ">=": op.ge, "<=": op.le, "==": op.eq}


def _ev(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 12:
            raise ValueError("pow too large")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    raise ValueError("disallowed expression")


def fmt_num(x):
    return str(int(x)) if isinstance(x, (int, float)) and float(x).is_integer() else str(x)


def evaluate(expr):
    """Return the result string for a span expression, or None if it cannot be computed."""
    try:
        if expr.startswith("date:"):
            base, n = expr[5:].split("+")
            return (dt.date.fromisoformat(base) + dt.timedelta(days=int(n))).isoformat()
        if expr.startswith("wd:"):
            return dt.date.fromisoformat(expr[3:]).strftime("%A")
        if expr.startswith("dur:"):
            a, b = expr[4:].split(",")
            return str(abs((dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days))
        if expr.startswith("conv:"):
            v, rest = expr[5:].split(" ", 1)
            a, b = rest.split(">")
            if (a, b) in _UNITS:
                return fmt_num(round(float(v) * _UNITS[(a, b)], 2))
            if (a, b) == ("c", "f"):
                return fmt_num(round(float(v) * 9 / 5 + 32, 2))
            if (a, b) == ("f", "c"):
                return fmt_num(round((float(v) - 32) * 5 / 9, 2))
            return None
        if expr.startswith("cmp:"):
            a, b = expr[4:].split(",")
            return "first" if float(a) > float(b) else "second" if float(a) < float(b) else "equal"
        if expr.startswith("sort:"):
            xs = expr[5:].split(",")
            return ",".join(sorted(xs, key=float))
        if expr.startswith("filter:"):
            cond, rest = expr[7:].split(":", 1)
            for sym, fn in sorted(_CMP_OPS.items(), key=lambda kv: -len(kv[0])):
                if cond.startswith(sym):
                    thr = float(cond[len(sym):])
                    xs = [x for x in rest.split(",") if fn(float(x), thr)]
                    return ",".join(xs) if xs else "none"
            return None
        if expr.startswith("count:"):
            return str(len(expr[6:]))
        for k, fn in (("mean:", lambda v: sum(v) / len(v)), ("max:", max), ("min:", min), ("sum:", sum)):
            if expr.startswith(k):
                r = fn([float(x) for x in expr[len(k):].split(",")]); return fmt_num(round(r, 4))
        if expr.startswith("gcd:"):
            import math; a, b = expr[4:].split(","); return str(math.gcd(int(a), int(b)))
        if expr.startswith("lcm:"):
            import math; a, b = expr[4:].split(","); return str(math.lcm(int(a), int(b)))
        if expr.startswith("prime:"):
            n = int(expr[6:]); return "yes" if n > 1 and all(n % d for d in range(2, int(n ** .5) + 1)) else "no"
        if expr.startswith("rev:"):
            return expr[4:][::-1]
        if expr.startswith("fact:"):
            import math; n = int(expr[5:]); return str(math.factorial(n)) if 0 <= n <= 20 else None
        if expr.startswith("sqrt:"):
            return fmt_num(round(float(expr[5:]) ** 0.5, 3))
        if expr.startswith("pchg:"):
            a, b = expr[5:].split(","); return fmt_num(round((float(b) - float(a)) / float(a) * 100, 2))
        if expr.startswith("words:"):
            return str(len(expr[6:].split()))
        return fmt_num(_ev(ast.parse(expr, mode="eval").body))
    except Exception:
        return None


def encode_span(tok, expr, result=None):
    """ids of <CALC> expr <EQ> [result] <ECALC>, one token per character inside."""
    ch = lambda s: [tok.token_to_id(c) if tok.token_to_id(c) is not None else tok.encode(c).ids[0] for c in s]
    ids = [CALC] + ch(expr) + [EQ]
    return ids + (ch(result) + [ECALC] if result is not None else [])


class CircuitLogitsProcessor:
    """HF logits processor (batch size 1): when the model emits <EQ> after a <CALC> span, force the exact result tokens + <ECALC>."""

    def __init__(self, tok):
        self.tok = tok
        self.queue, self.handled = [], -1

    def _chars(self, ids):
        return "".join(self.tok.id_to_token(i) or "" for i in ids)

    def __call__(self, input_ids, scores):
        import torch
        ids = input_ids[0].tolist()
        if not self.queue and ids and ids[-1] == EQ and len(ids) - 1 != self.handled:
            self.handled = len(ids) - 1
            try:
                s = max(i for i, t in enumerate(ids) if t == CALC)
                res = evaluate(self._chars(ids[s + 1:-1]).replace("Ġ", " "))
            except ValueError:
                res = None
            if res is not None:
                self.queue = encode_span(self.tok, "", res)[2:]          # result chars + <ECALC>
        if self.queue:
            t = self.queue.pop(0)
            forced = torch.full_like(scores, float("-inf")); forced[:, t] = 0.0
            return forced
        return scores
