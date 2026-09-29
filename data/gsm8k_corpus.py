"""Convert GSM8K (real, human-written, linguistically diverse grade-school math word problems) into
Q-164M's exact-circuit chat format. GSM8K's own `<<expr=result>>` calculator annotations are an almost
exact match for our `<CALC>expr<EQ>result<ECALC>` spans -- both exist so a small model never has to
compute arithmetic itself. This directly targets the "templated-eval generalization risk" flagged in
q-agent/PLAN.md: our own circuit_corpus.py generates diverse but still templated phrasing; GSM8K gives
real human phrasing for the same underlying skill.

Every extracted calc step is validated against OUR OWN evaluate() before inclusion -- if our arithmetic
evaluator doesn't reproduce GSM8K's stated result exactly, the whole example is dropped rather than
training on a tool call whose "ground truth" the executor would actually contradict at inference time.
"""
import re, sys, numpy as np
sys.path.insert(0, ".")
from tokenizers import Tokenizer
from datasets import load_dataset
from model.circuits_qagent import evaluate, encode_span, USER, MODEL, EOT, THINK, ETHINK

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
T = lambda s: tok.encode(s).ids
ANN = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def convert(question, answer):
    calc_ids = []
    for expr, expected in ANN.findall(answer):
        got = evaluate(expr)
        if got is None: return None
        try:
            if abs(float(got) - float(expected)) > 1e-6: return None
        except ValueError:
            if got.strip() != expected.strip(): return None
        calc_ids += encode_span(tok, expr, got)
    if not calc_ids: return None
    visible = ANN.sub("", answer)          # GSM8K writes "48/2 = <<48/2=24>>24 clips" -- the number
                                            # already appears right after the annotation in the source
                                            # text, so the annotation itself must be dropped, not replaced
    visible = re.sub(r"\n?####\s*(-?[\d,.]+)\s*$", "", visible).strip()  # drop the GSM8K trailer line
    if not visible: return None
    return [1, USER] + T(question) + [EOT, MODEL, THINK] + calc_ids + [ETHINK] + T(visible) + [EOT, 2]


if __name__ == "__main__":
    import os
    os.makedirs("data/circuits", exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")
    for split, name in [("train", "gsm8k_train"), ("test", "gsm8k_val")]:
        rows, dropped = [], 0
        for ex in ds[split]:
            r = convert(ex["question"], ex["answer"])
            if r is None: dropped += 1; continue
            rows.append(np.asarray(r, np.uint16))
        off = np.concatenate([[0], np.cumsum([len(x) for x in rows])]).astype(np.int64)
        np.savez(f"data/circuits/{name}.npz", t=np.concatenate(rows), o=off)
        print(name, len(rows), "rows kept,", dropped, "dropped (validation mismatch or no calc span),", int(off[-1]), "tokens")
