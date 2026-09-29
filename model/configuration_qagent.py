from transformers import PretrainedConfig


class QAgentConfig(PretrainedConfig):
    """Q-Agent: text-only ternary decoder, frozen fingerprint vocabulary (no trainable embedding table)
    plus a learned per-token correction covering the WHOLE vocabulary (unlike Q-Omni, there is no
    multimodal id range to special-case -- every id gets the same treatment). Sized ~164M for a from-
    scratch pretrain on a single V100 16GB: bigger than any prior text-only run in this project family,
    still comfortably single-GPU (a 111.8M sibling config measured ~11.6GB/16GB at this batch/seq)."""
    model_type = "qagent"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32832,           # 32768 text (existing byte-BPE tokenizer) + 64 reserved control ids
        code_bits=512,              # width of the frozen fingerprint of every token
        hidden_size=1280,
        intermediate_size=3520,
        num_hidden_layers=10,
        num_attention_heads=20,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=100000.0,
        initializer_range=0.02,
        qk_norm=True,
        attention_output_gate=True,
        quant_mode="ternary",       # "ternary" | "none" (fp16 ablation)
        act_quant=False,
        tie_word_embeddings=False,  # embedding is a frozen buffer, nothing to tie
        text_vocab_size=32768,      # ids < this are ordinary text tokens
        special_offset=32768,       # ids >= this are control tokens (dialogue turns, tool calls, ...)
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.code_bits = code_bits
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.qk_norm = qk_norm
        self.attention_output_gate = attention_output_gate
        self.quant_mode = quant_mode
        self.act_quant = act_quant
        self.text_vocab_size = text_vocab_size
        self.special_offset = special_offset
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                          tie_word_embeddings=tie_word_embeddings, **kwargs)
