"""QPACK1: single-file packed container for Q-Omni models (header, table of contents, 64-byte aligned blobs, mmap-friendly).

  magic  "QPACK1\0\0" (8 bytes) | count u32 | pad u32 | count x [name[56] NUL-padded, offset u64, length u64] | blobs, each aligned to 64 bytes

Contents: manifest.json, config.json, generation_config.json, tokenizer.json, selftest.json,
  ternary projections  -> <id>.tern5  (5 trits per byte = 1.6 bits/weight) + <id>.alpha (fp32 per-row scale)
  frozen fingerprints  -> codes.bits   (+-1 codes, 1 bit each)      trainable mm delta -> mm_delta.f16
  everything else      -> <id>.f16 / <id>.f32
Loading needs only numpy (+ torch to build tensors). Ternary weights are stored losslessly: t in {-1,0,+1} and the exact per-row alpha of the QAT forward pass."""
import json, mmap, os, re, struct
import numpy as np

MAGIC, ALIGN = b"QPACK1\0\0", 64
TERN_RE = re.compile(r"^model\.layers\.\d+\.(attn\.(q|k|v|o)_proj|mlp\.(w1|w2))\.weight$")
_LUT = None


def _lut():
    global _LUT
    if _LUT is None:
        v = np.arange(243, dtype=np.int32); _LUT = np.stack([(v // 3 ** i) % 3 for i in range(5)], 1).astype(np.int8)
    return _LUT


def pack_tern5(t):
    """t: int array in {-1,0,1}, any shape -> bytes (row-major, 5 trits per byte)"""
    d = (np.asarray(t, dtype=np.int8).reshape(-1) + 1).astype(np.uint8); pad = (-len(d)) % 5
    if pad: d = np.concatenate([d, np.ones(pad, np.uint8)])
    d = d.reshape(-1, 5).astype(np.uint16); return (d[:, 0] + 3 * d[:, 1] + 9 * d[:, 2] + 27 * d[:, 3] + 81 * d[:, 4]).astype(np.uint8).tobytes()


def unpack_tern5(buf, n):
    return (_lut()[np.frombuffer(buf, dtype=np.uint8)].reshape(-1)[:n] - 1).astype(np.int8)


def write_container(path, blobs):
    names = list(blobs); head = 16 + 72 * len(names); off = (head + ALIGN - 1) // ALIGN * ALIGN; ent = []
    for n in names:
        if len(n.encode()) > 55: raise ValueError("blob name too long: " + n)
        ent.append((n, off, len(blobs[n]))); off = (off + len(blobs[n]) + ALIGN - 1) // ALIGN * ALIGN
    with open(path, "wb") as o:
        o.write(MAGIC + struct.pack("<II", len(names), 0))
        for n, of, ln in ent: o.write(n.encode().ljust(56, b"\0") + struct.pack("<QQ", of, ln))
        for n, of, ln in ent:
            o.write(b"\0" * (of - o.tell())); o.write(blobs[n])
    return os.path.getsize(path)


class Container:
    def __init__(self, path):
        self.f = open(path, "rb"); self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        if self.mm[:8] != MAGIC: raise ValueError("not a QPACK1 container: " + str(path))
        n, _ = struct.unpack("<II", self.mm[8:16]); self.toc = {}
        for i in range(n):
            b = 16 + 72 * i; name = self.mm[b:b + 56].rstrip(b"\0").decode(); off, ln = struct.unpack("<QQ", self.mm[b + 56:b + 72]); self.toc[name] = (off, ln)

    def blob(self, name): off, ln = self.toc[name]; return self.mm[off:off + ln]
    def json(self, name): return json.loads(bytes(self.blob(name)).decode())
    def close(self): self.mm.close(); self.f.close()


def pack_state(state, cfg_dict, tokenizer_json=None, gen_cfg=None, dense="f16", selftest=None):
    """state: dict name -> torch tensor (a Q-Omni state_dict). Returns dict of blobs."""
    blobs, man, k = {}, [], 0
    dt = np.float16 if dense == "f16" else np.float32
    for name, w in state.items():
        a = w.detach().cpu().float().numpy(); ent = dict(name=name, shape=list(a.shape)); bid = f"w{k:04d}"; k += 1
        if TERN_RE.match(name) and cfg_dict.get("quant_mode") == "ternary":
            import torch   # mirror TernaryLinear.forward exactly (same op order), so the stored trits are the ones the QAT model computes with
            wf = w.detach().cpu().float(); al = wf.abs().mean(dim=1, keepdim=True).clamp(min=1e-5); alpha = al.numpy().astype(np.float32); t = torch.clamp(torch.round(wf / al), -1, 1).numpy().astype(np.int8)
            blobs[bid + ".tern5"] = pack_tern5(t); blobs[bid + ".alpha"] = alpha.reshape(-1).tobytes(); ent.update(kind="tern5", blob=bid + ".tern5", alpha=bid + ".alpha")
        elif name.endswith("model.codes"):
            blobs["codes.bits"] = np.packbits((a > 0).astype(np.uint8), axis=1).tobytes(); ent.update(kind="bits", blob="codes.bits")
        elif name.endswith(("mm_delta_in", "mm_delta_out")):
            tag = "mm_delta_in.f16" if name.endswith("mm_delta_in") else "mm_delta_out.f16"
            blobs[tag] = a.astype(np.float16).tobytes(); ent.update(kind="f16", blob=tag)
        elif name.endswith("logit_bias"):
            blobs[bid + ".f32"] = a.astype(np.float32).tobytes(); ent.update(kind="f32", blob=bid + ".f32")
        else:
            blobs[bid + (".f16" if dense == "f16" else ".f32")] = a.astype(dt).tobytes(); ent.update(kind="f16" if dense == "f16" else "f32", blob=bid + (".f16" if dense == "f16" else ".f32"))
        man.append(ent)
    blobs["manifest.json"] = json.dumps(dict(format="QPACK1", tensors=man, dense=dense)).encode(); blobs["config.json"] = json.dumps(cfg_dict).encode()
    if tokenizer_json: blobs["tokenizer.json"] = tokenizer_json if isinstance(tokenizer_json, bytes) else tokenizer_json.encode()
    if gen_cfg: blobs["generation_config.json"] = json.dumps(gen_cfg).encode()
    if selftest: blobs["selftest.json"] = json.dumps(selftest).encode()
    return blobs


def load_state(path):
    """QPACK1 container -> (state_dict of torch tensors ready for load_state_dict, config dict). Ternary weights come back as latent
    weights W' = t * alpha / f  (f = fraction of non-zeros in the row), for which the QAT forward pass (absmean re-quantisation)
    reproduces exactly t * alpha."""
    import torch
    c = Container(path); man = c.json("manifest.json"); cfg = c.json("config.json"); state = {}
    for e in man["tensors"]:
        shp = e["shape"]; n = int(np.prod(shp))
        if e["kind"] == "tern5":
            t = unpack_tern5(c.blob(e["blob"]), n).reshape(shp).astype(np.float32); alpha = np.frombuffer(c.blob(e["alpha"]), dtype=np.float32).reshape(-1, 1)
            f = np.maximum((t != 0).mean(1, keepdims=True), 1e-12); arr = np.where(t != 0, t * (alpha / f), 0.0).astype(np.float32)
        elif e["kind"] == "bits":
            arr = np.unpackbits(np.frombuffer(c.blob(e["blob"]), dtype=np.uint8).reshape(shp[0], -1), axis=1)[:, :shp[1]].astype(np.float32) * 2 - 1
        else:
            dtp = np.float16 if e["kind"] == "f16" else np.float32; arr = np.frombuffer(c.blob(e["blob"]), dtype=dtp).reshape(shp).astype(np.float32)
        tt = torch.from_numpy(np.ascontiguousarray(arr)); state[e["name"]] = tt.half() if e["name"].endswith("model.codes") else tt
    c.close(); return state, cfg
