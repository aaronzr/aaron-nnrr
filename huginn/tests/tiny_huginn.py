"""Tiny (64-dim) randomly initialized Huginn using the real model code and tokenizer, for tests.
"""

import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tiny(seed=0):
    from transformers import AutoTokenizer

    from tests.tiny_raven.raven_config_minimal import RavenConfig
    from tests.tiny_raven.raven_modeling_minimal import RavenForCausalLM

    torch.manual_seed(seed)
    config = RavenConfig(
        n_embd=64, n_heads=4, n_layers=8, block_size=1024, vocab_size=65536,
        intermediate_size=128, n_layers_in_recurrent_block=4, n_layers_in_prelude=2,
        n_layers_in_coda=2, mean_recurrence=8, torch_dtype="float32",
        init_values={"std": 0.02, "embed_scale": 8.0, "embedding": 0.02, "out_proj": 0.002},
    )
    model = RavenForCausalLM(config).float().eval()
    # Random init leaves the output weights near zero, which makes every token tie.
    # Nudge them so argmax and ranks are well defined in tests.
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2 and p.abs().max() == 0:
                p.normal_(0, 0.02)
    tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "tokenizer"))
    return model, tok
