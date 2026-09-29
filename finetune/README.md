# Fine-tuning Q-164M into Q-U-164M

Chat-SFT, then RL on a verifiable reward, on top of the base `Q-164M` pretrain checkpoint. Current release
is `v5` — see the [blog post](https://q-project-lm.github.io/blog-q164m.html) for the full v1-v5 story
(several earlier passes had a real multi-turn tool-calling bug; v5 is the fix).

## 1. Build the SFT data

Run each generator from the repo root (writes into `data/circuits/*.npz`):

```bash
python data/circuit_corpus.py
python data/gsm8k_corpus.py       # needs openai/gsm8k, downloaded via `datasets`
python data/smoltalk_corpus.py    # needs HuggingFaceTB/smoltalk's everyday-conversations subset
python data/multiturn_corpus.py   # synthetic multi-turn dialogues with conversational connectives
```

Then build the actual SFT training set (assistant-only loss mask, the v5 mix):

```bash
python data/sft_build_qagent.py   # -> data/sft/{train,val}.npz
```

## 2. Chat-SFT

```bash
python finetune/sft.py --init runs/q164m_pretrain/final.pt --out runs/qu164m_sft --steps 12000
```

Tracks a multi-turn tool-calling probe at every eval (not just val_loss) and saves the best checkpoint by
that metric to `runs/qu164m_sft/best.pt` — val_loss alone can keep improving while multi-turn reliability
quietly regresses (this happened in an earlier pass; see the blog post's v4 section), so don't rely on
`final.pt` without checking `best.pt` too.

## 3. RL

GRPO-style, rule-based verifiable reward from the model's own deterministic tool executor
(`model/circuits_qagent.py`'s `evaluate()`) — no judge model, no preference pairs. The executor either
agrees with the gold answer or it doesn't, which is a strictly better reward signal than anything a judge
model could approximate here.

```bash
python finetune/rl.py --init runs/qu164m_sft/final.pt --out runs/qu164m_rl \
    --lr 2e-6 --kl_coef 0.1 --steps 200
```

`lr`/`kl_coef`/`steps` are the values that trained stably in v4 and v5 — an earlier attempt at `lr=2e-5,
kl_coef=0.02` collapsed the policy within ~100 steps, and `steps=300` let KL blow up past step ~200 in one
run. Also saves a `best.pt` tracked by eval reward, for the same reason as SFT.

## 4. Verify before trusting either checkpoint

Loss and reward curves are not proof the model actually works — see the self-verification principle in the
blog post. At minimum, run a fixed single-turn probe (arithmetic, dates, primality, sorting) and an 8-turn
multi-turn conversation using natural phrasing (not the bare, templated style the synthetic corpus itself
uses — that gap is exactly what caused the v1-v4 bugs). `../examples/chat.py` in this repo has a runnable
version of both.
