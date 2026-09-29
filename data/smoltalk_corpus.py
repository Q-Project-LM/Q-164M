"""Real multi-turn dialogue from HuggingFaceTB/smoltalk's `everyday-conversations` subset (Apache-2.0,
Magpie-generated but curated -- short, natural, on-topic exchanges). Deliberately NOT using smoltalk's
`apigen-80k` subset: that one trains JSON-schema function calling (`<tool_call>{...}</tool_call>` against
an arbitrary in-context tool signature), which is the implicit/schema-based tool-calling paradigm that
q-agent/RESEARCH.md found needs ~270M-1B+ params to work reliably -- mixing it in would fight the whole
point of Q-164M's explicit-trigger + deterministic-executor design, not support it.
No <CALC> spans here -- this corpus is purely for multi-turn dialogue fluency/diversity, complementing our
own templated chit-chat and GSM8K's math-word-problem phrasing diversity."""
import sys, numpy as np
sys.path.insert(0, ".")
from tokenizers import Tokenizer
from datasets import load_dataset
from model.circuits_qagent import USER, MODEL, EOT

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
T = lambda s: tok.encode(s).ids


def convert(messages):
    ids = [1]
    for m in messages:
        if m["role"] == "system": continue
        role = USER if m["role"] == "user" else MODEL
        ids += [role] + T(m["content"]) + [EOT]
    ids += [2]
    return np.asarray(ids, np.uint16)


if __name__ == "__main__":
    import os
    os.makedirs("data/circuits", exist_ok=True)
    ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")
    for split, name in [("train", "smoltalk_train"), ("test", "smoltalk_val")]:
        rows = [convert(ex["messages"]) for ex in ds[split] if len(ex["messages"]) >= 2]
        off = np.concatenate([[0], np.cumsum([len(x) for x in rows])]).astype(np.int64)
        np.savez(f"data/circuits/{name}.npz", t=np.concatenate(rows), o=off)
        print(name, len(rows), "rows,", int(off[-1]), "tokens")
