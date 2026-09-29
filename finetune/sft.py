"""Q-U-164M chat SFT on top of the Q-164M pretrain checkpoint, assistant-only loss.
usage: python sft.py --init runs/q164m_pretrain/final.pt --out runs/qu164m_sft --steps 3000 (single GPU, fp16 autocast)
Data: data/sft/{train,val}.npz (see data/sft_build_qagent.py)."""
import argparse, math, os, sys, time, numpy as np, torch
sys.path.insert(0, ".")
from model.configuration_qagent import QAgentConfig
from model.modeling_qagent import QAgentForCausalLM
from model.circuits_qagent import USER, MODEL, EOT, EQ, ECALC, CircuitLogitsProcessor
from tokenizers import Tokenizer

# Multi-turn tool-use probe run at every eval, alongside val_loss: val_loss alone missed a real regression
# in the qu164m_sft_v4 run, where multi-turn correctness peaked around step 6000 then visibly degraded by
# step 9500 (model abandoning tool calls turn 5+) while val_loss kept improving the whole time -- val_loss
# only measures next-token prediction on the training distribution, not whether the model still calls
# tools correctly many turns into a conversation. best.pt is saved whenever this score improves, so a
# late-training regression (like v4's) doesn't silently overwrite the best checkpoint the run produced.
_tok = Tokenizer.from_file("tokenizer/tokenizer.json")
_TURNS = [("Hi! Quick question: what's 12 times 12?", "144"),
          ("Thanks. Now what's 200 minus 37?", "163"),
          ("Got it. Can you sort these: 5, 2, 9, 1?", "1,2,5,9"),
          ("Nice. What's 15% of 80?", "12"),
          ("OK. Is 53 a prime number?", "yes"),
          ("Cool. What's 1000 divided by 8?", "125"),
          ("One more: what's 45 plus 78?", "123"),
          ("Last one: sort these: 3, 30, 3.5, 0.3.", "0.3,3,3.5,30")]


def _calc_result(ids):
    if EQ not in ids: return None
    i = ids.index(EQ) + 1
    j = ids.index(ECALC, i) if ECALC in ids[i:] else len(ids)
    return "".join(_tok.decode([t]) for t in ids[i:j]).strip()


@torch.no_grad()
def multiturn_score(model, dev):
    model.eval()
    history = [1]
    correct = 0
    for q, expect in _TURNS:
        history += [USER] + _tok.encode(q).ids + [EOT, MODEL]
        cp = CircuitLogitsProcessor(_tok); x = torch.tensor([history]).to(dev); past = None; out = []
        for _ in range(96):
            with torch.autocast(dev, dtype=torch.float16):
                o = model(x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            sc = cp(torch.tensor([history + out]), o.logits[:, -1].float())
            t = int(sc.argmax()); out.append(t); x = torch.tensor([[t]]).to(dev)
            if t == EOT: break
        got = _calc_result(out)
        correct += got is not None and got.replace(" ", "") == expect.replace(" ", "")
        history += out
    model.train()
    return correct / len(_TURNS)


p = argparse.ArgumentParser()
p.add_argument("--init", required=True); p.add_argument("--out", required=True); p.add_argument("--steps", type=int, default=3000)
p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--warmup", type=int, default=100)
p.add_argument("--micro", type=int, default=8); p.add_argument("--accum", type=int, default=4); p.add_argument("--seq", type=int, default=1024)
p.add_argument("--eval_every", type=int, default=200); p.add_argument("--save_every", type=int, default=500)
p.add_argument("--resume", type=int, default=1)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

ck = torch.load(a.init, map_location="cpu"); cfg = QAgentConfig(**ck["cfg"])
m = QAgentForCausalLM(cfg); print("init:", m.load_state_dict(ck["model"], strict=False)); m = m.to(dev)
m.model.delta_in.requires_grad_(False); m.model.delta_out.requires_grad_(False)  # keep the pretrained fingerprint corrections frozen; SFT only adapts the transformer body + readout

ztr = np.load("data/sft/train.npz"); Ttr, Mtr, Otr = ztr["t"], ztr["m"], ztr["o"]; ntr = len(Otr) - 1
zva = np.load("data/sft/val.npz"); Tva, Mva, Ova = zva["t"], zva["m"], zva["o"]; nva = len(Ova) - 1
rng = np.random.default_rng(0); vrng = np.random.default_rng(1234)


def make_batch(T, M, O, n, rr):
    X = np.zeros((a.micro, a.seq), np.int64); Y = np.full((a.micro, a.seq), -100, np.int64)
    for b in range(a.micro):
        pos = 0
        while pos < a.seq:
            k = int(rr.integers(n)); r = T[O[k]:O[k + 1]].astype(np.int64); mk = M[O[k]:O[k + 1]]
            L = min(len(r), a.seq - pos)
            X[b, pos:pos + L] = r[:L]; Y[b, pos:pos + L] = np.where(mk[:L] == 1, r[:L], -100); pos += L
    return torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev)


def batch(): return make_batch(Ttr, Mtr, Otr, ntr, rng)
def vbatch(): return make_batch(Tva, Mva, Ova, nva, vrng)


dec = [q for q in m.parameters() if q.requires_grad and q.ndim >= 2]; nod = [q for q in m.parameters() if q.requires_grad and q.ndim < 2]
opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.01}, {"params": nod, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95), fused=True)
sc = torch.amp.GradScaler("cuda")
lr_at = lambda s: a.lr * (s + 1) / a.warmup if s < a.warmup else a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))

step = 0; ck_path = f"{a.out}/last.pt"; best_mt_score = [-1.0]
if a.resume and os.path.exists(ck_path):
    c = torch.load(ck_path, map_location="cpu"); m.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sc.load_state_dict(c["scaler"]); step = c["step"]; print("resumed", step, flush=True)

log = open(f"{a.out}/train.log", "a"); t0 = time.time()
m.train()
while step < a.steps:
    for g in opt.param_groups: g["lr"] = lr_at(step)
    tl = 0.0
    for _ in range(a.accum):
        x, y = batch()
        with torch.autocast("cuda", dtype=torch.float16): loss = m(x, labels=y).loss / a.accum
        sc.scale(loss).backward(); tl += loss.item()
    sc.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True)
    step += 1
    if step % 20 == 0 or step == 1:
        s = f"step={step} loss={tl:.4f} gn={gn:.2f} lr={lr_at(step):.2e} tok/s={step*a.micro*a.accum*a.seq/(time.time()-t0):.0f} elapsed={time.time()-t0:.0f}s"
        print(s, flush=True); log.write(s + "\n"); log.flush()
    if step % a.eval_every == 0 or step == a.steps:
        m.eval(); vl = 0.0
        with torch.no_grad():
            for _ in range(20):
                x, y = vbatch()
                with torch.autocast("cuda", dtype=torch.float16): vl += m(x, labels=y).loss.item()
        vl /= 20; s = f"EVAL step={step} val_loss={vl:.4f} val_ppl={math.exp(vl):.2f}"; print(s, flush=True); log.write(s + "\n"); log.flush()
        mt = multiturn_score(m, dev)
        s2 = f"EVALMT step={step} multiturn_score={mt:.3f}"; print(s2, flush=True); log.write(s2 + "\n"); log.flush()
        if mt > best_mt_score[0]:
            best_mt_score[0] = mt
            torch.save({"model": m.state_dict(), "cfg": cfg.to_dict(), "step": step, "multiturn_score": mt}, f"{a.out}/best.pt")
            print(f"[best] step={step} multiturn_score={mt:.3f}", flush=True)
        m.train()
    if step % a.save_every == 0 or step == a.steps:
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "scaler": sc.state_dict(), "step": step, "cfg": cfg.to_dict()}, ck_path + ".tmp"); os.replace(ck_path + ".tmp", ck_path)
torch.save({"model": m.state_dict(), "cfg": cfg.to_dict(), "step": step}, f"{a.out}/final.pt"); print("done", flush=True)
