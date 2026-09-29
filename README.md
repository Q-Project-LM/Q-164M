<div align="center">
  <img src="Q_Logo.svg" width="72" alt="Q Project logo">
  <h1>Q-164M</h1>
  <h3>Fine-tuning and inference code for a 164.6M-parameter ternary text model with exact tool calls</h3>

  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" alt="License"></a>
    <a href="https://huggingface.co/q-project/Q-164M"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Hugging%20Face-yellow?style=for-the-badge" alt="Hugging Face"></a>
    <a href="https://q-project-lm.github.io/blog-q164m.html"><img src="https://img.shields.io/badge/Blog-How%20it%20works-2F6FD0?style=for-the-badge" alt="Blog"></a>
  </p>
</div>

> [!IMPORTANT]
> This is a research artifact, not a product. It is small and often factually wrong outside its trained
> tool-calling domain. See the [blog post](https://q-project-lm.github.io/blog-q164m.html) and the
> [QBench leaderboard](https://huggingface.co/spaces/q-project/QBench) for honest results before drawing
> conclusions from any headline number.

## What's here, what isn't

This repo has the **fine-tuning recipe and inference code** for Q-164M / Q-U-164M — not the pretrain
pipeline (that stays internal) and not the weights (those are on the [Hub](https://huggingface.co/q-project/Q-164M),
where `trust_remote_code=True` pulls in the small architecture files below automatically).

```
model/          the architecture reference: modeling_qagent.py, configuration_qagent.py, circuits_qagent.py
data/           data generators used by the SFT mix (circuit_corpus.py, gsm8k_corpus.py, smoltalk_corpus.py,
                multiturn_corpus.py) and the script that assembles them (sft_build_qagent.py)
finetune/       chat-SFT (sft.py) + rule-based-RL (rl.py) — see finetune/README.md for the full recipe
examples/       chat.py (minimal inference example) and verify.py (the hands-on multi-turn stress test
                that found and fixed a real bug across several fine-tune passes)
run.py          interactive CLI: `python run.py --instruct`
tokenizer/      tokenizer.json — needed to run and fine-tune the model
```

## What this model is

Q-164M is a 164,648,648-parameter decoder-only model: every token gets a **frozen, random ±1
"fingerprint"** (no trainable embedding table at all) plus a small learned correction, split into untied
`delta_in`/`delta_out` covering the whole 32,832-token vocabulary. The transformer body is **ternary
`{-1,0,+1}·α`** end-to-end from step 1 of training (BitNet b1.58-style QAT). A small deterministic
**circuit** intercepts `<CALC>expr<EQ>` spans during generation and force-decodes the exact result from a
fixed evaluator (`model/circuits_qagent.py`) instead of hoping the network memorized arithmetic — the
model's job is only ever to decide *when* and *which* tool to call, never to compute the answer itself.
Pretrained from scratch on a single 16GB V100: 200,000 steps, ~92.4 hours, 13.1B tokens, zero NaNs.

**Q-U-164M** is the same checkpoint, fine-tuned in two stages specifically for dialogue — chat-SFT, then
RL on a verifiable reward from the model's own tool executor. See [`finetune/README.md`](finetune/README.md)
for the recipe, and the [blog post](https://q-project-lm.github.io/blog-q164m.html) for the full v1-v5
story (several earlier passes had a real multi-turn tool-calling bug, root-caused and fixed in v5).

## Quickstart

```bash
pip install -r requirements.txt
python run.py --instruct
```

```
> What is 91 divided by 7?
91 divided by 7 is 13.
> Thanks. Now sort these: 9, 3, 7, 1.
9, 3, 7, 1 sorted is 1, 3, 7, 9.
```

Pass `--show_tool_calls` to see the `[expr=result]` span the model actually generated for each reply, or
see [`examples/chat.py`](examples/chat.py) for the same thing as a plain script using `transformers`
directly.

## How it compares to other small models

Measured with [QBench](https://github.com/Q-Project-LM/QBench) — no model gets forced-correct tool
execution in this comparison, including this one, so scores are comparable across architectures. See the
[leaderboard](https://huggingface.co/spaces/q-project/QBench) for the live table and the full methodology.

| model | params | single-turn tool-use | multi-turn tool-use | ARC-Easy | TruthfulQA MC1 |
|---|---|---|---|---|---|
| GPT-2 | 124M | 0% | 0% | 42% | 26% |
| SmolLM2-360M-Instruct | 360M | 50% | 50% | 50% | 18% |
| Qwen2.5-0.5B-Instruct | 500M | 45% | 50% | 62% | 24% |
| **Q-U-164M** | **164M** | 35% | 25% | 43% | **26%** |

Q-U-164M doesn't win outright here, and shouldn't be expected to — it's a third the size of Qwen2.5-0.5B,
trained on one V100. What it shows: a frozen-embedding, ternary-weight, single-GPU model stays within reach
of models several times its size and training budget. Deployed normally — with its own deterministic
circuit executor correcting whichever calculation it decides to make — its actual tool-call accuracy is
much higher than these unassisted numbers; the architecture's whole premise is that the network never has
to get the arithmetic right itself, only decide when to ask.

## Architecture

```
vocab_size          = 32,832   (32,768 text (byte-BPE) + 64 reserved control ids)
code_bits           = 512      (width of the frozen per-token fingerprint)
hidden_size         = 1,280
intermediate_size   = 3,520    (plain 2-matrix SiLU MLP, not SwiGLU)
num_hidden_layers   = 10
attention           = GQA, 20 query heads / 4 KV heads, head_dim 64, QK-norm, per-head sigmoid output gate
positional encoding = RoPE (theta = 1e5)
quantization        = ternary {-1,0,+1}*alpha, per-output-row absmean scale, straight-through estimator
parameters          = 164,648,648
```

## License

Code: Apache-2.0 (see `LICENSE`). Third-party data used by the fine-tuning recipe keeps its own license —
see `NOTICE`.
