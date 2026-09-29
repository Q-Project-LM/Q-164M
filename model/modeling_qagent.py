import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_qagent import QAgentConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class TernaryLinear(nn.Linear):
    """Weight is re-quantised to {-1,0,+1}*alpha (per-output-row absmean alpha) on every forward; gradients pass
    straight through to the latent full-precision master weight (BitNet b1.58-style QAT, trained this way from step 1)."""

    def __init__(self, in_features, out_features, act_quant=False):
        super().__init__(in_features, out_features, bias=False)
        self.act_quant = act_quant

    def ternary(self):
        wf = self.weight.float()
        alpha = wf.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        return torch.clamp(torch.round(wf / alpha), -1, 1), alpha

    def forward(self, x):
        wf = self.weight.float()
        alpha = wf.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        wq = torch.clamp(torch.round(wf / alpha), -1, 1) * alpha
        w = (wf + (wq - wf).detach()).to(x.dtype)
        if self.act_quant:
            s = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
            x = x + (torch.clamp(torch.round(x / s), -127, 127) * s - x).detach()
        return F.linear(x, w)


def make_linear(cfg, i, o):
    return TernaryLinear(i, o, cfg.act_quant) if cfg.quant_mode == "ternary" else nn.Linear(i, o, bias=False)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class RotaryEmbedding(nn.Module):
    """inv_freq is computed on the fly (a non-persistent buffer would be lost/garbled by meta-device init in from_pretrained)."""

    def __init__(self, dim, theta):
        super().__init__()
        self.dim, self.theta = dim, theta

    def forward(self, position_ids, dtype):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=position_ids.device).float() / self.dim))
        f = position_ids[..., None].float() * inv[None, None, :]
        e = torch.cat((f, f), dim=-1)
        return e.cos().to(dtype), e.sin().to(dtype)


class QAgentAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        d = cfg.hidden_size
        self.q_proj = make_linear(cfg, d, self.nh * self.hd)
        self.k_proj = make_linear(cfg, d, self.nkv * self.hd)
        self.v_proj = make_linear(cfg, d, self.nkv * self.hd)
        self.o_proj = make_linear(cfg, self.nh * self.hd, d)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()
        self.out_gate = nn.Linear(d, self.nh, bias=True) if cfg.attention_output_gate else None  # per-head sigmoid gate

    def forward(self, x, cos, sin, cache=None, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        c, s = cos[:, None], sin[:, None]
        q, k = q * c + rotate_half(q) * s, k * c + rotate_half(k) * s
        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and k.shape[2] == T))
        if self.out_gate is not None:
            o = o * torch.sigmoid(self.out_gate(x)).transpose(1, 2).unsqueeze(-1)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, self.nh * self.hd))


class QAgentMLP(nn.Module):
    """Plain two-matrix SiLU feed-forward (d -> d_ff -> d), both ternary."""

    def __init__(self, cfg):
        super().__init__()
        self.w1 = make_linear(cfg, cfg.hidden_size, cfg.intermediate_size)
        self.w2 = make_linear(cfg, cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class QAgentLayer(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn, self.mlp = QAgentAttention(cfg, i), QAgentMLP(cfg)

    def forward(self, x, cos, sin, cache=None, mask=None):
        x = x + self.attn(self.n1(x), cos, sin, cache, mask)
        return x + self.mlp(self.n2(x))


class QAgentPreTrainedModel(PreTrainedModel):
    config_class = QAgentConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["QAgentLayer"]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, self.config.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, RMSNorm):
            nn.init.ones_(m.weight)
        elif isinstance(m, QAgentModel):
            nn.init.zeros_(m.delta_in); nn.init.zeros_(m.delta_out)


class QAgentModel(QAgentPreTrainedModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # frozen fingerprint table: one +-1 code of cfg.code_bits per token (buffer, NOT a parameter; 0
        # trainable embedding params). Unlike Q-Omni, there is no multimodal id range: every id in the
        # vocabulary (text tokens AND control/dialogue/tool tokens alike) gets the SAME treatment -- one
        # shared frozen fingerprint plus one learned per-id correction, split into separate in/out deltas
        # (Q-Omni's `learned_text_deltas` ablation showed tying embedding and readout roles into a single
        # correction vector measurably hurts both; untying is the validated default here, not an add-on).
        self.register_buffer("codes", torch.sign(torch.randn(cfg.vocab_size, cfg.code_bits)).to(torch.float16), persistent=True)
        self.delta_in = nn.Parameter(torch.zeros(cfg.vocab_size, cfg.code_bits))
        self.delta_out = nn.Parameter(torch.zeros(cfg.vocab_size, cfg.code_bits))
        self.w_in = nn.Linear(cfg.code_bits, cfg.hidden_size, bias=False)   # dense fp io
        self.layers = nn.ModuleList([QAgentLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.post_init()

    def table_in(self, dtype):
        return (self.codes.float() + self.delta_in.float()).to(dtype)

    def table_out(self, dtype):
        return (self.codes.float() + self.delta_out.float()).to(dtype)

    def embed(self, ids):
        dt = torch.float16 if torch.is_autocast_enabled() else self.w_in.weight.dtype
        return self.w_in(F.embedding(ids, self.table_in(dt)))

    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=False, attention_mask=None):
        B, T = input_ids.shape
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past, past + T, device=input_ids.device)[None].expand(B, -1)
        x = self.embed(input_ids)
        cos, sin = self.rotary(position_ids, x.dtype)
        mask = None
        if past > 0 or attention_mask is not None:
            S = past + T
            qi = torch.arange(past, S, device=x.device)[:, None]
            mask = (torch.arange(S, device=x.device)[None, :] <= qi)[None, None]
            if attention_mask is not None:
                mask = mask & attention_mask[:, None, None, :S].bool()
        for layer in self.layers:
            x = layer(x, cos, sin, past_key_values, mask)
        return self.norm(x), past_key_values


class QAgentForCausalLM(QAgentPreTrainedModel, GenerationMixin):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = QAgentModel(cfg)
        self.w_out = nn.Linear(cfg.hidden_size, cfg.code_bits, bias=False)  # readout: hidden -> fingerprint space
        self.register_buffer("logit_bias", torch.zeros(cfg.vocab_size), persistent=True)  # frozen unigram log-prior
        self.post_init()

    def logits_of(self, h):
        z = self.w_out(h)
        return (z @ self.model.table_out(z.dtype).t()) / math.sqrt(self.config.code_bits) + self.logit_bias.to(z.dtype)

    def forward(self, input_ids=None, labels=None, position_ids=None, past_key_values=None, use_cache=None,
                attention_mask=None, logits_to_keep=0, **kw):
        h, past_key_values = self.model(input_ids, position_ids, past_key_values, bool(use_cache), attention_mask)
        if labels is None:
            if logits_to_keep:
                h = h[:, -logits_to_keep:]
            return CausalLMOutputWithPast(logits=self.logits_of(h), past_key_values=past_key_values)
        hs, ys = h[:, :-1].reshape(-1, h.size(-1)), labels[:, 1:].reshape(-1)

        def ce(hc, yc):  # chunked + checkpointed: full (tokens x vocab) logits are never materialised
            return F.cross_entropy(self.logits_of(hc).float(), yc, ignore_index=-100, reduction="sum")

        tot = hs.new_zeros((), dtype=torch.float32)
        for i in range(0, hs.size(0), 4096):
            tot = tot + torch.utils.checkpoint.checkpoint(ce, hs[i:i + 4096], ys[i:i + 4096], use_reentrant=False)
        return CausalLMOutputWithPast(loss=tot / (ys != -100).sum().clamp(min=1), past_key_values=past_key_values)
