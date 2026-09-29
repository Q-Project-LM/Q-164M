"""Hands-on verification: an 8-task single-turn probe plus an 8-turn multi-turn stress test using natural
conversational phrasing (not the bare, templated style the synthetic training corpus itself uses -- that
gap is what caused several real bugs across earlier fine-tune passes; see the blog post).

A low loss or a good RL reward is not proof a model actually works -- always check generation directly.

usage:
  python examples/verify.py                              # loads q-project/Q-164M, instruct/ subfolder
  python examples/verify.py --ckpt runs/qu164m_rl/final.pt  # or a local checkpoint from finetune/
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from model.circuits_qagent import USER, MODEL, EOT, CircuitLogitsProcessor

PROBE = [
    "Hello, who are you?",
    "What is 347 times 86?",
    "I have 3 books and I bought 5 more. How many books do I have now?",
    "My bill is 240 dollars. What is 15 percent of that?",
    "What date is 45 days after December 20, 2026?",
    "Is 97 a prime number?",
    "Sort these numbers from smallest to largest: 9, 3, 7, 1.",
    "Explain what photosynthesis is in one sentence.",
]

MULTI_TURN = [
    "Hi! Quick question: what's 12 times 12?",
    "Thanks. Now what's 200 minus 37?",
    "Got it. Can you sort these: 5, 2, 9, 1?",
    "Nice. What's 15% of 80?",
    "OK. Is 53 a prime number?",
    "Cool. What's 1000 divided by 8?",
    "One more: what's 45 plus 78?",
    "Last one: sort these: 3, 30, 3.5, 0.3.",
]


def load(ckpt_path):
    from tokenizers import Tokenizer
    from model.configuration_qagent import QAgentConfig
    from model.modeling_qagent import QAgentForCausalLM

    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu")
        model = QAgentForCausalLM(QAgentConfig(**ck["cfg"]))
        model.load_state_dict(ck["model"], strict=False)
        print(f"loaded {ckpt_path} (step {ck.get('step')})")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("q-project/Q-164M", subfolder="instruct",
                                                       trust_remote_code=True)
        print("loaded q-project/Q-164M (instruct/)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tok, device


def show(tok, ids):
    out = []
    for t in ids:
        if t == 32773:
            out.append("[calc:")
        elif t == 32774:
            out.append("=")
        elif t == 32775:
            out.append("]")
        elif t < 32768:
            out.append(tok.decode([t]))
    return "".join(out)


@torch.no_grad()
def reply(model, tok, device, history_ids, n=110):
    cp = CircuitLogitsProcessor(tok)
    x = torch.tensor([history_ids]).to(device)
    past, out = None, []
    for _ in range(n):
        o = model(x, past_key_values=past, use_cache=True)
        past = o.past_key_values
        sc = cp(torch.tensor([history_ids + out]), o.logits[:, -1].float())
        t = int(sc.argmax())
        out.append(t)
        x = torch.tensor([[t]]).to(device)
        if t == EOT:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    a = p.parse_args()
    model, tok, device = load(a.ckpt)

    print("\n=== 8-task single-turn probe ===")
    for q in PROBE:
        ids = [1, USER] + tok.encode(q).ids + [EOT, MODEL]
        out = reply(model, tok, device, ids)
        print(f"Q: {q}\nA: {show(tok, out)[:220]}\n")

    print("\n=== 8-turn multi-turn stress test (natural phrasing) ===")
    history = [1]
    for i, q in enumerate(MULTI_TURN, 1):
        history += [USER] + tok.encode(q).ids + [EOT, MODEL]
        out = reply(model, tok, device, history)
        history += out
        print(f"--- turn {i} ---\nQ: {q}\nA: {show(tok, out)[:220]}\n")


if __name__ == "__main__":
    main()
