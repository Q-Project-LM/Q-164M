"""Synthetic chat rows that teach the model WHEN to call a circuit (it writes the span; the circuit supplies the result).
Row: <bos><USER> question <EOT><MODEL><THINK><CALC>expr<EQ>result<ECALC><ETHINK> answer <EOT><eos>   (chit-chat rows have no span)."""
import random, sys, datetime as dt, numpy as np
sys.path.insert(0, ".")
from tokenizers import Tokenizer
from model.circuits_qagent import *
tok = Tokenizer.from_file("tokenizer/tokenizer.json"); T = lambda s: tok.encode(s).ids
NAMES = ["books", "apples", "pens", "cards", "coins", "stickers", "marbles", "cookies"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
def row(q, expr, ans_fmt):
    res = evaluate(expr) if expr else None
    a = ans_fmt.format(r=res)
    think = [THINK] + encode_span(tok, expr, res) + [ETHINK] if expr else []
    return [1, USER] + T(q) + [EOT, MODEL] + think + T(a) + [EOT, 2]
def row_multi(q, expr_fns, ans_fmt):
    """Agentic multi-step tool use: several <CALC>...<EQ>...<ECALC> spans in ONE <THINK> block, where
    each expr_fn(prior_results) can reference the results of earlier steps -- the model has to decide
    to call a second tool using the first tool's output, not just answer after one lookup."""
    results, span_ids = [], []
    for fn in expr_fns:
        expr = fn(results); res = evaluate(expr)
        if res is None: return None
        results.append(res); span_ids += encode_span(tok, expr, res)
    a = ans_fmt.format(*results)
    return [1, USER] + T(q) + [EOT, MODEL, THINK] + span_ids + [ETHINK] + T(a) + [EOT, 2]
def gen(r):
    k = r.randrange(24)
    if k == 0:
        a, b, o = r.randint(2, 999), r.randint(2, 99), r.choice("+-*")
        w = {"+": "plus", "-": "minus", "*": "times"}[o]
        return row(r.choice([f"What is {a} {w} {b}?", f"Calculate {a} {o} {b}.", f"{a} {o} {b} = ?"]), f"{a}{o}{b}", f"{a} {w} {b} is {{r}}.")
    if k == 1:
        a, b = r.randint(2, 40), r.randint(2, 40); n = r.choice(NAMES)
        if r.random() < .5: return row(f"I have {a} {n} and I bought {b} more. How many {n} do I have now?", f"{a}+{b}", f"You have {{r}} {n}.")
        a, b = a + b, b; return row(f"I have {a} {n} and I gave away {b}. How many {n} are left?", f"{a}-{b}", f"You have {{r}} {n} left.")
    if k == 2:
        a, p = r.randint(10, 900), r.choice([5, 10, 12, 15, 20, 25, 30, 40, 50])
        return row(f"My bill is {a} dollars. What is {p} percent of that?", f"{a}*{p}/100", "{r} dollars.")
    if k == 3:
        d = dt.date(2025, 1, 1) + dt.timedelta(days=r.randint(0, 1500)); n = r.randint(1, 120)
        return row(f"What date is {n} days after {MONTHS[d.month-1]} {d.day}, {d.year}?", f"date:{d.isoformat()}+{n}", "__DATE__" + str(n)) if False else \
               (lambda e: [1, USER] + T(f"What date is {n} days after {MONTHS[d.month-1]} {d.day}, {d.year}?") + [EOT, MODEL, THINK] + encode_span(tok, f"date:{d.isoformat()}+{n}", e) + [ETHINK] + T(f"{MONTHS[int(e[5:7])-1]} {int(e[8:])}, {e[:4]}.") + [EOT, 2])(evaluate(f"date:{d.isoformat()}+{n}"))
    if k == 4:
        d = dt.date(2025, 1, 1) + dt.timedelta(days=r.randint(0, 2000))
        return row(f"What day of the week is {MONTHS[d.month-1]} {d.day}, {d.year}?", f"wd:{d.isoformat()}", f"{MONTHS[d.month-1]} {d.day}, {d.year} is a {{r}}.")
    if k == 5:
        (a, b) = r.choice([("km", "mi"), ("mi", "km"), ("kg", "lb"), ("lb", "kg"), ("c", "f"), ("f", "c")]); v = r.randint(1, 300)
        nm = {"km": "kilometers", "mi": "miles", "kg": "kilograms", "lb": "pounds", "c": "degrees Celsius", "f": "degrees Fahrenheit"}
        return row(f"Convert {v} {nm[a]} to {nm[b]}.", f"conv:{v} {a}>{b}", f"{v} {nm[a]} is about {{r}} {nm[b]}.")
    if k == 6:
        a, b = round(r.uniform(1, 20), r.choice([1, 2])), round(r.uniform(1, 20), r.choice([1, 2]))
        return row(f"Which is bigger, {a} or {b}?", f"cmp:{a},{b}", "The {r} number is bigger." )
    if k == 7:
        xs = [str(r.randint(1, 99)) for _ in range(r.randint(3, 7))]
        return row("Sort these numbers from smallest to largest: " + ", ".join(xs) + ".", "sort:" + ",".join(xs), "{r}.")
    if k == 8:
        w = r.choice(["banana", "computer", "elephant", "science", "library", "mountain", "rainbow", "keyboard"])
        return row(f"How many letters are in the word {w}?", f"count:{w}", f"The word {w} has {{r}} letters.")
    if 9 <= k <= 17:
        j = k - 9
        if j == 0:
            xs = [r.randint(1, 100) for _ in range(r.randint(3, 6))]; o = r.choice(["mean", "max", "min", "sum"]); w = {"mean": "the average of", "max": "the largest of", "min": "the smallest of", "sum": "the sum of"}[o]
            return row(f"What is {w} {', '.join(map(str, xs))}?", f"{o}:" + ",".join(map(str, xs)), f"{w.capitalize()} {', '.join(map(str, xs))} is {{r}}.")
        if j == 1:
            a, b = r.randint(2, 60), r.randint(2, 60); return row(f"What is the greatest common divisor of {a} and {b}?", f"gcd:{a},{b}", f"The greatest common divisor of {a} and {b} is {{r}}.")
        if j == 2:
            a, b = r.randint(2, 30), r.randint(2, 30); return row(f"What is the least common multiple of {a} and {b}?", f"lcm:{a},{b}", f"The least common multiple of {a} and {b} is {{r}}.")
        if j == 3:
            n = r.randint(2, 400); return row(f"Is {n} a prime number?", f"prime:{n}", f"{n}: {{r}}.")
        if j == 4:
            w = r.choice(["hello", "computer", "banana", "science", "rainbow", "window", "garden", "puzzle"]); return row(f"Spell {w} backwards.", f"rev:{w}", f"{w} backwards is {{r}}.")
        if j == 5:
            n = r.randint(1, 12); return row(f"What is {n} factorial?", f"fact:{n}", f"{n} factorial is {{r}}.")
        if j == 6:
            n = r.choice([4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 2, 3, 5, 7, 10, 50]); return row(f"What is the square root of {n}?", f"sqrt:{n}", f"The square root of {n} is about {{r}}.")
        if j == 7:
            a, b = r.randint(10, 500), r.randint(10, 500); return row(f"The price went from {a} dollars to {b} dollars. What is the percent change?", f"pchg:{a},{b}", "The change is {r} percent.")
        s2 = " ".join(r.choice(["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "runs", "far", "away"]) for _ in range(r.randint(3, 9)))
        return row(f"How many words are in this sentence: {s2}?", f"words:{s2}", "It has {r} words.")
    if k == 18:
        d1 = dt.date(2024, 1, 1) + dt.timedelta(days=r.randint(0, 1200)); d2 = d1 + dt.timedelta(days=r.randint(1, 500))
        q1 = f"{MONTHS[d1.month-1]} {d1.day}, {d1.year}"; q2 = f"{MONTHS[d2.month-1]} {d2.day}, {d2.year}"
        return row(r.choice([f"How many days are between {q1} and {q2}?", f"How many days is it from {q1} to {q2}?"]), f"dur:{d1.isoformat()},{d2.isoformat()}", "{r} days.")
    if k == 20:
        p, d = r.randint(20, 900), r.choice([5, 10, 15, 20, 25, 30, 40])
        return row_multi(f"An item costs {p} dollars with a {d} percent discount. What is the final price?",
                          [lambda res: f"{p}*{d}/100", lambda res: f"{p}-{res[0]}"],
                          "The discount is {0} dollars, so the final price is {1} dollars.")
    if k == 21:
        a, b, c = r.randint(5, 80), r.randint(5, 80), 0
        c = a + b + r.randint(5, 50)
        return row_multi(f"I bought a {r.choice(NAMES)[:-1]} for {a} dollars and another for {b} dollars, and paid with a {c} dollar bill. How much change do I get?",
                          [lambda res: f"{a}+{b}", lambda res: f"{c}-{res[0]}"],
                          "The total was {0} dollars, so the change is {1} dollars.")
    if k == 19:
        xs = [r.randint(1, 100) for _ in range(r.randint(4, 8))]; op, thr = r.choice([(">", r.randint(10, 80)), ("<", r.randint(20, 90)), (">=", r.randint(10, 80)), ("<=", r.randint(20, 90))])
        nm = {">": "greater than", "<": "less than", ">=": "at least", "<=": "at most"}[op]
        return row(f"From this list, keep only the numbers {nm} {thr}: {', '.join(map(str, xs))}.", f"filter:{op}{thr}:" + ",".join(map(str, xs)), "{r}.")
    q = r.choice(["Hi!", "Hello, who are you?", "Tell me a short joke about computers.", "What can you do?", "Thanks!"])
    a = {"Hi!": "Hello! How can I help you?", "Hello, who are you?": "I am Q, a small local model.", "Tell me a short joke about computers.": "Why did the computer go to the doctor? Because it had a virus.", "What can you do?": "I can chat, and I hand exact calculations and a few other tasks to a built-in circuit.", "Thanks!": "You are welcome."}[q]
    return [1, USER] + T(q) + [EOT, MODEL] + T(a) + [EOT, 2]
if __name__ == "__main__":
    import os; os.makedirs("data/circuits", exist_ok=True)
    for name, n, seed in [("train", 600000, 11), ("val", 4000, 12)]:
        r = random.Random(seed); rows = [np.asarray(gen(r), np.uint16) for _ in range(n)]
        off = np.concatenate([[0], np.cumsum([len(x) for x in rows])]).astype(np.int64)
        np.savez(f"data/circuits/{name}.npz", t=np.concatenate(rows), o=off); print(name, n, "rows", int(off[-1]), "tokens")
