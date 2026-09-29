"""Q-U-164M RL fine-tune: GRPO-style, rule-based verifiable reward via the project's own deterministic
executor (model.circuits_qagent.evaluate) -- not DPO on static offline pairs. Rationale: Q-164M's whole
tool mechanism already produces a machine-checkable correct/incorrect signal for free (the executor either
agrees with the gold answer or it doesn't), which is a much better fit than preference pairs assembled by
a judge model. It directly targets the one skill the pretrain run never converged on (`sort:` -- see the
blog-q164m.html UPD): sort-family prompts are oversampled in the RL prompt pool the same way they were in
SFT.

Rollout: no KV cache (sequences here are short -- recomputing the full prefix every step is simpler and
avoids padding/cache bugs, see RESEARCH note in this file's history). Per-sequence CircuitLogitsProcessor
still runs during rollout, so the executor forces the correct digits after <EQ> exactly like real
inference; those forced positions are excluded from the policy-gradient loss (they are an environment
response, not a policy action) -- tracked via `forced[i]` at generation time, not inferred after the fact.

usage: python rl.py --init runs/qu164m_sft/final.pt --out runs/qu164m_rl --steps 400
"""
import argparse, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from tokenizers import Tokenizer
from model.configuration_qagent import QAgentConfig
from model.modeling_qagent import QAgentForCausalLM
from model.circuits_qagent import USER, MODEL, EOT, THINK, ETHINK, CALC, EQ, ECALC, evaluate, CircuitLogitsProcessor

p = argparse.ArgumentParser()
p.add_argument("--init", required=True); p.add_argument("--out", required=True); p.add_argument("--steps", type=int, default=400)
p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--warmup", type=int, default=20)
p.add_argument("--prompts_per_step", type=int, default=12); p.add_argument("--group_size", type=int, default=6)
p.add_argument("--max_new_tokens", type=int, default=96); p.add_argument("--temperature", type=float, default=0.9)
p.add_argument("--kl_coef", type=float, default=0.02); p.add_argument("--eval_every", type=int, default=25); p.add_argument("--save_every", type=int, default=50)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
ck = torch.load(a.init, map_location="cpu"); cfg = QAgentConfig(**ck["cfg"])
policy = QAgentForCausalLM(cfg); print("policy init:", policy.load_state_dict(ck["model"], strict=False)); policy = policy.to(dev).train()
policy.model.delta_in.requires_grad_(False); policy.model.delta_out.requires_grad_(False)
ref = QAgentForCausalLM(cfg); ref.load_state_dict(ck["model"], strict=False); ref = ref.to(dev).eval()
for q in ref.parameters(): q.requires_grad_(False)


# ---- RL prompt pool: (prompt_ids, gold_kind or None, gold_result or None) ----
def expr_kind(chars):
    return chars.split(":", 1)[0] if ":" in chars else "arith"


def chars_between(row, i, j):
    """decode the raw per-character token ids row[i:j] back to text (span-internal tokens are one char each)."""
    return "".join(tok.id_to_token(int(t)) or "" for t in row[i:j]).replace("Ġ", " ")


def build_pool(name, gold=True, max_rows=None):
    z = np.load(f"data/circuits/{name}.npz"); t, o = z["t"], z["o"]
    rows = []
    n = len(o) - 1 if max_rows is None else min(max_rows, len(o) - 1)
    for i in range(n):
        r = t[o[i]:o[i + 1]].astype(int).tolist()
        j = r.index(MODEL) if MODEL in r else None
        if j is None:
            continue
        prompt = r[:j + 1]
        gkind = gresult = None
        if gold and CALC in r and r.count(CALC) == 1:  # keep RL reward simple: single-tool-call rows only
            s = r.index(CALC); e = r.index(EQ, s)
            expr = chars_between(r, s + 1, e)
            gkind = expr_kind(expr)
            f = r.index(ECALC, e)
            gresult = chars_between(r, e + 1, f)
        elif gold:
            continue  # multi-step chains and no-tool rows aren't part of the tool-reward pool
        rows.append((prompt, gkind, gresult))
    return rows


circuit_pool = build_pool("train")
gsm8k_pool = build_pool("gsm8k_train")
sort_pool = [r for r in circuit_pool if r[1] == "sort"]
other_pool = [r for r in circuit_pool if r[1] != "sort"]
dialogue_pool = build_pool("smoltalk_train", gold=False, max_rows=2000)
print(f"RL pools: circuit {len(circuit_pool)} ({len(sort_pool)} sort), gsm8k {len(gsm8k_pool)}, dialogue {len(dialogue_pool)}", flush=True)

POOL_WEIGHTS = [(sort_pool, 0.35), (other_pool, 0.30), (gsm8k_pool, 0.25), (dialogue_pool, 0.10)]
rng = random.Random(0)


def sample_prompts(n):
    out = []
    for _ in range(n):
        pool, _ = rng.choices(POOL_WEIGHTS, weights=[w for _, w in POOL_WEIGHTS])[0]
        out.append(rng.choice(pool))
    return out


# ---- reward ----
def reward_of(gen_ids, gkind, gresult):
    r = 0.0
    end = gen_ids.index(EOT) if EOT in gen_ids else None
    r += 0.3 if end is not None else -0.3
    body = gen_ids[:end] if end is not None else gen_ids
    if gkind is not None:
        if CALC in body and EQ in body[body.index(CALC):]:
            s = body.index(CALC); e = body.index(EQ, s)
            kind = expr_kind(chars_between(body, s + 1, e))
            r += 1.0 if kind == gkind else -0.2
            if ECALC in body[e:]:
                f = body.index(ECALC, e)
                got = chars_between(body, e + 1, f)
                r += 1.0 if got == gresult else -0.2
        else:
            r -= 0.5  # should have called a tool, didn't
    else:
        uniq = len(set(body)) / max(1, len(body))
        r += 0.4 * min(1.0, uniq * 2)  # cheap anti-degenerate-repetition signal for the no-gold dialogue anchor rows
    return r


@torch.no_grad()
def rollout(prompts):
    """prompts: list of (ids, gkind, gresult); returns per-sequence token ids, free-token mask, rewards."""
    B = len(prompts)
    seqs = [list(p[0]) for p in prompts]
    done = [False] * B
    procs = [CircuitLogitsProcessor(tok) for _ in range(B)]
    free_mask_tail = [[] for _ in range(B)]
    for _ in range(a.max_new_tokens):
        if all(done):
            break
        L = max(len(s) for s in seqs)
        X = torch.zeros(B, L, dtype=torch.long, device=dev)
        attn = torch.zeros(B, L, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            X[i, L - len(s):] = torch.tensor(s, device=dev); attn[i, L - len(s):] = 1
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = policy(input_ids=X, position_ids=pos, attention_mask=attn, logits_to_keep=1).logits[:, -1].float()
        logits = logits / a.temperature
        for i in range(B):
            if done[i]:
                continue
            pre_empty = not procs[i].queue
            sc = procs[i](torch.tensor([seqs[i]]), logits[i:i + 1].clone())
            probs = F.softmax(sc[0], dim=-1)
            t = int(torch.multinomial(probs, 1))
            seqs[i].append(t)
            free_mask_tail[i].append(pre_empty)
            if t == EOT:
                done[i] = True
    rewards = [reward_of(seqs[i][len(prompts[i][0]):], prompts[i][1], prompts[i][2]) for i in range(B)]
    return seqs, free_mask_tail, rewards, [len(p[0]) for p in prompts]


def seq_logp_and_kl(model, ref_model, seqs, plens, free_masks):
    """teacher-forced logp of the actually-sampled tokens under `model`, restricted to free (non-forced)
    positions, plus a per-token KL(model || ref) estimate on those same positions."""
    L = max(len(s) for s in seqs)
    X = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
    attn = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
    lossmask = torch.zeros(len(seqs), L, dtype=torch.bool, device=dev)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.tensor(s, device=dev); attn[i, :len(s)] = 1
        for k, free in enumerate(free_masks[i]):
            if free:
                lossmask[i, plens[i] + k] = True  # position of the token predicted at step k (0-indexed after prompt)
    tgt = X[:, 1:].clamp(min=0)

    # Chunk over the batch dim: materializing (batch, seq, vocab) logits for the whole rollout batch at once
    # (32832-wide vocab, policy AND ref, both needing a live autograd graph on the policy side) OOMs a 16GB V100
    # well before batch=72 -- gather down to (batch, seq) log-probs per chunk instead, never holding more than
    # CHUNK rows of full-vocab logits in memory at a time.
    CHUNK = 8
    lp_parts, rlp_parts = [], []
    for i in range(0, len(seqs), CHUNK):
        xb, ab, tb = X[i:i + CHUNK], attn[i:i + CHUNK], tgt[i:i + CHUNK]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=xb, attention_mask=ab).logits[:, :-1].float()
            with torch.no_grad():
                rlogits = ref_model(input_ids=xb, attention_mask=ab).logits[:, :-1].float()
        lp_parts.append(F.log_softmax(logits, dim=-1).gather(-1, tb.unsqueeze(-1)).squeeze(-1))
        rlp_parts.append(F.log_softmax(rlogits, dim=-1).gather(-1, tb.unsqueeze(-1)).squeeze(-1))
    lp, rlp = torch.cat(lp_parts, 0), torch.cat(rlp_parts, 0)
    m = lossmask[:, 1:].float()
    kl = ((lp - rlp) * m)  # per-token KL estimate (log-ratio at the sampled token, standard PPO/GRPO approximation)
    return lp, rlp, kl, m


dec = [q for q in policy.parameters() if q.requires_grad and q.ndim >= 2]; nod = [q for q in policy.parameters() if q.requires_grad and q.ndim < 2]
opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.0}, {"params": nod, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95), fused=True)
lr_at = lambda s: a.lr * (s + 1) / a.warmup if s < a.warmup else a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))
log = open(f"{a.out}/train.log", "a"); t0 = time.time()
best_eval_reward = [float("-inf")]

for step in range(1, a.steps + 1):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    prompts = []
    for p_ in sample_prompts(a.prompts_per_step):
        prompts += [p_] * a.group_size
    seqs, free_masks, rewards, plens = rollout(prompts)
    rewards_t = torch.tensor(rewards, device=dev)
    adv = torch.zeros_like(rewards_t)
    for gstart in range(0, len(prompts), a.group_size):
        grp = rewards_t[gstart:gstart + a.group_size]
        adv[gstart:gstart + a.group_size] = (grp - grp.mean()) / (grp.std() + 1e-4)

    lp, _, kl, m = seq_logp_and_kl(policy, ref, seqs, plens, free_masks)
    ntok = m.sum().clamp(min=1)
    pg_loss = -((adv.unsqueeze(-1) * lp) * m).sum() / ntok
    kl_loss = (kl * m).sum() / ntok
    loss = pg_loss + a.kl_coef * kl_loss

    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    del seqs, free_masks, lp, kl, m, adv

    if step % 5 == 0 or step == 1:
        s = (f"step={step} loss={loss.item():.4f} pg={pg_loss.item():.4f} kl={kl_loss.item():.4f} "
             f"reward_mean={rewards_t.mean().item():.3f} reward_std={rewards_t.std().item():.3f} "
             f"gn={gn:.2f} lr={lr_at(step):.2e} elapsed={time.time()-t0:.0f}s")
        print(s, flush=True); log.write(s + "\n"); log.flush()
    torch.cuda.empty_cache()  # rollout lengths vary step to step; without this, the caching allocator
                              # fragments badly enough on a 16GB card to OOM well before real usage is high
    if step % a.eval_every == 0 or step == a.steps:
        eval_prompts = sample_prompts(6) * 1
        eseqs, _, erewards, _ = rollout([(p_[0], p_[1], p_[2]) for p_ in eval_prompts])
        eval_reward_mean = sum(erewards) / len(erewards)
        s = f"EVAL step={step} eval_reward_mean={eval_reward_mean:.3f}"
        print(s, flush=True); log.write(s + "\n"); log.flush()
        # save the best-eval checkpoint separately from last/final: a run that ends on an unstable step
        # (KL/grad-norm spike late in training, seen in the qu164m_rl_v3 run) otherwise leaves no way to
        # recover the better intermediate policy once last.pt has been overwritten past that point.
        if eval_reward_mean > best_eval_reward[0]:
            best_eval_reward[0] = eval_reward_mean
            torch.save({"model": policy.state_dict(), "cfg": cfg.to_dict(), "step": step,
                        "eval_reward_mean": eval_reward_mean}, f"{a.out}/best.pt")
            print(f"[best] step={step} eval_reward_mean={eval_reward_mean:.3f}", flush=True)
    if step % a.save_every == 0 or step == a.steps:
        torch.save({"model": policy.state_dict(), "cfg": cfg.to_dict(), "step": step}, f"{a.out}/last.pt")

torch.save({"model": policy.state_dict(), "cfg": cfg.to_dict(), "step": a.steps}, f"{a.out}/final.pt")
print("done", flush=True)
