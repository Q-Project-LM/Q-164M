"""Minimal inference example: load Q-U-164M from the Hub and have an exact-tool-call conversation with it.
usage: python examples/chat.py
Needs: pip install transformers torch tokenizers
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from model.circuits_qagent import USER, MODEL, EOT, CircuitLogitsProcessor

MODEL_ID = "q-project/Q-U-164M"  # the chat-SFT + RL fine-tune; use "q-project/Q-164M" for the base pretrain model

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
model = model.cuda().eval() if torch.cuda.is_available() else model.eval()
device = next(model.parameters()).device


@torch.no_grad()
def reply(history_ids, max_new_tokens=96):
    """Greedy-decode one assistant turn. CircuitLogitsProcessor is what makes the tool calls exact: once the
    model writes <CALC>expr<EQ>, it forces the deterministic result instead of letting the network guess."""
    cp = CircuitLogitsProcessor(tok)
    x = torch.tensor([history_ids]).to(device)
    past, out = None, []
    for _ in range(max_new_tokens):
        logits = model(x, past_key_values=past, use_cache=True)
        past = logits.past_key_values
        scores = cp(torch.tensor([history_ids + out]), logits.logits[:, -1].float())
        t = int(scores.argmax())
        out.append(t)
        x = torch.tensor([[t]]).to(device)
        if t == EOT:
            break
    return out


def show(ids):
    """Render control tokens as readable brackets instead of the raw special-token surface forms."""
    parts = []
    for t in ids:
        if t == 32773:  # <CALC>
            parts.append("[")
        elif t == 32774:  # <EQ>
            parts.append("=")
        elif t == 32775:  # <ECALC>
            parts.append("] ")
        elif t >= 32768:
            continue
        else:
            parts.append(tok.decode([t]))
    return "".join(parts)


if __name__ == "__main__":
    history = [1]  # <bos>
    turns = [
        "Hi! Quick question: what's 347 times 86?",
        "Thanks. Now sort these: 9, 3, 7, 1.",
        "One more: is 97 a prime number?",
    ]
    for q in turns:
        history += [USER] + tok.encode(q).ids + [EOT, MODEL]
        out = reply(history)
        history += out
        print(f"user: {q}")
        print(f"model: {show(out)}\n")
