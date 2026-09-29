"""Synthetic MULTI-turn chat rows, built by chaining several independent circuit_corpus.gen() turns into
one conversation. Exists because the original SFT mix's multi-turn coverage was almost entirely real
SmolTalk data (a small slice of the mix, oversampled 3x but still a small fraction of 696k rows) while the
circuit/GSM8K corpora -- the great majority of SFT rows -- are single-turn only. That imbalance is the
direct, confirmed cause of a real bug: base Q-164M (and, to a lesser extent, the first SFT/RL pass) drops
its own tool-calling ability and starts repeating itself by the SECOND turn of a chat-formatted
conversation, because it essentially never saw a tool call anywhere but the first turn during training.

v1 of this corpus used 2-4 turns and measurably fixed turns 2-5 (verified by hand: base model failed at
turn 2, the v1-trained model held turns 2-5 correctly) -- but a 7-turn stress test then showed the SAME
class of failure reappearing at turns 6-7, exactly the out-of-distribution pattern this corpus exists to
fix, just pushed later because training data never went past 4 turns. v2 extends coverage to 2-8 turns,
with a dedicated long-tail slice (5-8 turns) built separately so it isn't diluted by shorter samples."""
import random, sys
import numpy as np

sys.path.insert(0, ".")
from circuit_corpus import gen, T  # reuses the exact same generators/format as the single-turn corpus

# v4->v5: hands-on testing found the model handles 7/8 turns correctly when questions are bare/templated
# (exactly how this corpus phrased every turn) but collapses to 2/8 when questions use natural
# conversational connectives ("Thanks. Now...", "Got it. Can you...") -- because this corpus NEVER
# includes them, so the model has never seen "acknowledge + tool call" in the same training example. Real
# users write exactly this way, so this isn't a cosmetic gap. Splicing connectives onto turns 2+ (most of
# the time, not always, so bare style is still seen too) teaches "handle small talk AND still call the
# tool" directly, instead of hoping real chat data (which has no tool calls at all) would somehow transfer
# the skill -- that's what v3 tried and it just taught the model to imitate chatty prose instead.
CONNECTIVES = [
    "Thanks. ", "Thanks! ", "Got it. ", "OK. ", "Ok, ", "Nice. ", "Cool. ", "Great, thanks. ",
    "Perfect. ", "Alright. ", "Sure, thanks. ", "One more question: ", "One more: ",
    "Another one: ", "Also, ", "And ", "Next, ", "Now, ", "Quick follow-up: ",
    "Last one: ", "Last question: ", "Can I ask one more thing? ", "Hmm ok, ", "Awesome, thanks. ",
]


def build(n, seed, min_turns, max_turns, p_connective=0.7):
    r = random.Random(seed)
    rows = []
    for _ in range(n):
        k = r.randint(min_turns, max_turns)
        turns = [gen(r) for _ in range(k)]  # each: [1, USER, ..., EOT, MODEL, ..., EOT, 2]
        body = []
        for i, t in enumerate(turns):
            t = t[1:-1]  # strip this turn's own <bos>/<eos>; only the whole conversation gets one of each
            if i > 0 and r.random() < p_connective:
                t = [t[0]] + T(r.choice(CONNECTIVES)) + t[1:]  # t[0] is the <USER> tag; splice after it
            body += t
        rows.append(np.asarray([1] + body + [2], np.uint16))
    return rows


def save(name, rows):
    off = np.concatenate([[0], np.cumsum([len(x) for x in rows])]).astype(np.int64)
    np.savez(f"data/circuits/{name}.npz", t=np.concatenate(rows), o=off)
    print(name, len(rows), "rows,", int(off[-1]), "tokens")


if __name__ == "__main__":
    import os
    os.makedirs("data/circuits", exist_ok=True)
    # short+medium slice (2-8 turns, broad coverage) and a dedicated long-tail slice (5-8 turns only, so
    # the exact range that previously failed isn't just a thin minority of a 2-8 uniform draw)
    save("multiturn_train", build(150000, 21, 2, 8))
    save("multiturn_val", build(3000, 22, 2, 8))
    save("multiturn_long_train", build(80000, 23, 5, 8))
    save("multiturn_long_val", build(1500, 24, 5, 8))
