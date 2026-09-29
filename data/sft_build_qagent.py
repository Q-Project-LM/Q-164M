"""Build the Q-U-164M chat-SFT dataset from the corpora already used for the Q-164M pretrain mix
(data/circuits/{train,val,gsm8k_train,gsm8k_val,smoltalk_train,smoltalk_val}.npz). Unlike the pretrain
mixture (where circ_w=0.15 keeps dialogue rare so the base model mostly still learns to write fluent
text), SFT needs every row to be a well-formed conversation and needs an assistant-only loss mask -- these
npz files carry no mask, so this script derives one generically: mask=1 for every token strictly after a
MODEL token up to and including the EOT that closes that turn (covers single-turn circuit/GSM8K rows and
multi-turn SmolTalk rows alike), mask=0 everywhere else (USER turns, BOS, trailing EOS).

Mix rationale ("best SFT" given what's on disk, see RESEARCH.md for why apigen-style JSON tool-calling was
rejected): weight AWAY from the templated circuit corpus (already seen heavily at pretrain, adds little new
signal) and TOWARD real dialogue phrasing (SmolTalk) and real math phrasing (GSM8K), since SFT's job here is
fluency + robustness, not introducing tool-calling from scratch. Sort-family circuit rows (the pretrain
run's one clearly unconverged skill -- see blog-q164m.html UPD) are additionally oversampled within the
circuit slice so SFT gives disproportionate exposure to the model's known weak point.

v3 added ultrachat_{train,val}/smoltalk_extra_{train,val} (real human/assistant conversations, no tool
calls) to multiturn_long_{train,val}, hoping real phrasing diversity would carry over -- hands-on testing
showed the opposite: the model imitated real chat's "let's think step by step" prose INSTEAD OF calling
tools from turn 3 onward.

v5 (this version, after v4 also regressed on the same axis) found the ACTUAL root cause via a controlled
A/B: the same v4 checkpoint scored 7/8 on an 8-turn stress test phrased in bare/templated style (matching
multiturn_corpus.py's own turns exactly) but only 2/8 on the identical test phrased with natural
conversational connectives ("Thanks. Now...", "Got it. Can you..."). multiturn_corpus.py never included
such connectives -- every turn was a bare standalone question -- so the model had never seen "acknowledge
+ still call the tool" in one example. Real users type exactly that way. Fix: multiturn_corpus.py now
splices natural connectives onto turns 2+ most of the time (see CONNECTIVES there), teaching the tool-call
skill directly in the phrasing that actually gets used, instead of hoping unrelated real chat data would
transfer it. ultrachat/smoltalk_extra are DROPPED from this mix entirely -- they were solving a problem
(no natural phrasing) that connective-splicing now solves without importing a zero-tool-calls style."""
import os, random, sys
import numpy as np

sys.path.insert(0, ".")
from tokenizers import Tokenizer
from model.circuits_qagent import MODEL, EOT, CALC

tok = Tokenizer.from_file("tokenizer/tokenizer.json")


def mask_row(row):
    """mask[i]=1 iff token i lies strictly after some MODEL token, up to and including the EOT that closes
    that turn. Handles multi-turn rows (repeated USER/MODEL/EOT) generically."""
    n = len(row)
    m = np.zeros(n, np.uint8)
    i = 0
    while i < n:
        if row[i] == MODEL:
            j = i + 1
            while j < n and row[j] != EOT:
                j += 1
            j = min(j, n - 1)  # include the closing EOT itself, never run past the row
            m[i + 1:j + 1] = 1
            i = j + 1
        else:
            i += 1
    return m


def is_sort_row(row):
    if CALC not in row:
        return False
    s = row.index(CALC)
    chars = "".join(tok.id_to_token(int(t)) or "" for t in row[s + 1:s + 8]).replace("Ġ", " ")
    return chars.startswith("sort:")


def load_rows(name):
    z = np.load(f"data/circuits/{name}.npz")
    t, o = z["t"], z["o"]
    return [t[o[i]:o[i + 1]].astype(np.int64).tolist() for i in range(len(o) - 1)]


def build(split, out_name, seed):
    rng = random.Random(seed)
    circuit = load_rows(split)
    gsm8k = load_rows(f"gsm8k_{split}")
    smoltalk = load_rows(f"smoltalk_{split}")
    mt_name = "multiturn_train" if split == "train" else "multiturn_val"
    multiturn = load_rows(mt_name)
    multiturn_long = load_rows(f"multiturn_long_{split}")

    is_sort = [is_sort_row(r) for r in circuit]
    sort_rows = [r for r, s in zip(circuit, is_sort) if s]
    other_rows = [r for r, s in zip(circuit, is_sort) if not s]
    print(f"[{split}] circuit rows: {len(circuit)} total, {len(sort_rows)} sort-family, "
          f"gsm8k {len(gsm8k)}, smoltalk {len(smoltalk)}, multiturn {len(multiturn)}, "
          f"multiturn_long {len(multiturn_long)}")

    rows = []
    rows += other_rows
    rows += sort_rows * 4     # oversample the model's known weak point
    rows += gsm8k * 2         # real math phrasing, upweighted vs templated circuit rows
    rows += smoltalk * 3      # real multi-turn dialogue, upweighted for fluency (SFT's main job)
    # 2-8 turn synthetic conversations (see multiturn_corpus.py): every SFT row above except SmolTalk is
    # single-turn, so a model trained on this mix alone essentially never sees a tool call on turn 2+ --
    # the confirmed, root cause of the base/first-SFT model dropping tool calls and repeating itself after
    # the first reply. Weighted to ~30% of the final pool, not just a token addition.
    rows += multiturn * 2
    # dedicated 5-8-turn slice, oversampled on its own: the 7-turn stress test on v2 (trained on 2-4-turn
    # data only) showed the fix's failure boundary just moves to turns 6-7, so the exact previously-failing
    # range needs real density, not just being a thin tail of the broader 2-8 draw above.
    rows += multiturn_long * 3
    rng.shuffle(rows)

    masks = [mask_row(r) for r in rows]
    keep = [i for i, mk in enumerate(masks) if mk.sum() > 0]
    rows, masks = [rows[i] for i in keep], [masks[i] for i in keep]

    off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    t = np.concatenate([np.asarray(r, np.uint16) for r in rows])
    m = np.concatenate(masks)
    os.makedirs("data/sft", exist_ok=True)
    np.savez(f"data/sft/{out_name}.npz", t=t, m=m, o=off)
    trained_tok = int(m.sum())
    print(f"[{split}] -> data/sft/{out_name}.npz : {len(rows)} rows, {int(off[-1])} tokens, "
          f"{trained_tok} assistant-loss tokens ({100 * trained_tok / max(1, off[-1]):.1f}%)")


if __name__ == "__main__":
    build("train", "train", seed=0)
    build("val", "val", seed=1)
