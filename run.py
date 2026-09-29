"""Minimal interactive chat CLI.
usage:
  python run.py                                 # base pretrain model, plain text continuation
  python run.py --instruct                      # Q-U-164M (chat-SFT + RL), exact tool calls
  python run.py --instruct --ckpt runs/x/final.pt  # a local checkpoint from finetune/ instead of the Hub
  python run.py --instruct --show_tool_calls    # print the raw [expr=result] span for each reply
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from model.circuits_qagent import USER, MODEL, EOT, CircuitLogitsProcessor


def load(instruct, ckpt_path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    if ckpt_path:
        from model.configuration_qagent import QAgentConfig
        from model.modeling_qagent import QAgentForCausalLM
        ck = torch.load(ckpt_path, map_location="cpu")
        model = QAgentForCausalLM(QAgentConfig(**ck["cfg"]))
        model.load_state_dict(ck["model"], strict=False)
    else:
        from transformers import AutoModelForCausalLM
        repo_id = "q-project/Q-U-164M" if instruct else "q-project/Q-164M"
        model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tok, device


def render(tok, ids, show_tool_calls):
    text_parts, span = [], []
    in_span = False
    for t in ids:
        if t == 32773:  # <CALC>
            in_span = True
            span = ["["]
        elif t == 32775:  # <ECALC>
            in_span = False
            if show_tool_calls:
                text_parts.append("".join(span) + "]")
        elif t == 32774:  # <EQ>
            if in_span:
                span.append("=")
        elif t < 32768:
            (span if in_span else text_parts).append(tok.decode([t]))
    return "".join(text_parts).strip()


@torch.no_grad()
def reply(model, tok, device, history_ids, max_new_tokens=110):
    cp = CircuitLogitsProcessor(tok)
    x = torch.tensor([history_ids]).to(device)
    past, out = None, []
    for _ in range(max_new_tokens):
        o = model(x, past_key_values=past, use_cache=True)
        past = o.past_key_values
        scores = cp(torch.tensor([history_ids + out]), o.logits[:, -1].float())
        t = int(scores.argmax())
        out.append(t)
        x = torch.tensor([[t]]).to(device)
        if t == EOT:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruct", action="store_true", help="load Q-U-164M instead of the base pretrain model")
    p.add_argument("--ckpt", default=None, help="load a local .pt checkpoint instead of the Hub")
    p.add_argument("--show_tool_calls", action="store_true")
    a = p.parse_args()

    model, tok, device = load(a.instruct, a.ckpt)
    print(f"loaded ({'instruct' if a.instruct else 'base'}{', ' + a.ckpt if a.ckpt else ''}). Ctrl-C to exit.\n")

    history = [1]
    while True:
        try:
            q = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        history += [USER] + tok.encode(q).ids + [EOT, MODEL]
        out = reply(model, tok, device, history)
        history += out
        print(render(tok, out, a.show_tool_calls))


if __name__ == "__main__":
    main()
