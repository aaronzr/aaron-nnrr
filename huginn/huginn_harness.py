"""Helpers for running Huginn-0125: prompt building, per-loop state capture,
R4 interventions (patching / steering), and logit/coda lens decoding.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

MODEL_ID = "tomg-group-umd/huginn-0125"
# Same commit Lu et al. pinned, so the replication is apples to apples.
REVISION = "2a364bd96e3eaa831be324f7c1f9e74892e4e594"
SYSTEM_PROMPT = "You are a concise and helpful assistant. Always return only the final answer straightway."

R4 = 3  # index of the last recurrent block in model.transformer.core_block


def load_huginn(device="cuda", dtype=torch.bfloat16, revision=REVISION):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, torch_dtype=dtype, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model.eval().to(device)
    return model, tokenizer


def build_input_ids(tokenizer, fewshot, question, device):
    """4-shot prompt in Huginn's chat format, matching Lu et al. Appendix A.2."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in fewshot:
        messages.append({"role": "user", "content": ex["context"]})
        messages.append({"role": "Huginn", "content": ex["completion"].strip()})
    messages.append({"role": "user", "content": question})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    return ids.to(device)


@dataclass
class Intervention:
    """
    What to do to the R4 output at a given loop (1-indexed).

    mode="replace": overwrite with `state` (activation patching). `state` must have the
                    same shape as the R4 output, i.e. [1, seq_len, d_model].
    mode="add":     add `state` (a steering vector, shape [d_model] or broadcastable).
    positions:      token positions to touch. None means every position.
    """
    loop: int
    state: torch.Tensor
    mode: str = "replace"
    positions: Optional[list] = None


@dataclass
class RunOutput:
    logits: torch.Tensor                 # [vocab] logits for the first answer token (float32)
    pred_id: int
    pred_text: str
    block_states: dict = field(default_factory=dict)  # block index -> list over loops of [1, L, d]


class Runner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.d_model = model.config.n_embd
        self._digit_ids = {}
        for d in range(10):
            ids = set()
            for form in (str(d), "Ġ" + str(d)):  # "7" and " 7" (Ġ7)
                tid = tokenizer.convert_tokens_to_ids(form)
                if tid is not None and tid != tokenizer.unk_token_id:
                    ids.add(tid)
            self._digit_ids[str(d)] = sorted(ids)
        self.the_ids = [t for t in (tokenizer.convert_tokens_to_ids("the"),) if t is not None]


    def make_init_state(self, seq_len, kind="seed", seed=0, scale=1.0):
        """
        Huginn starts its loop from random noise, and by default it draws NEW noise on
        every forward pass. Passing an explicit state makes runs reproducible and lets us
        study the starting state on purpose.

        kind="seed":  Huginn's own init (truncated normal * embed scale), drawn from `seed`
        kind="zero":  all zeros (Huginn's own deterministic option)
        scale:        multiply the noise, e.g. 0.5 or 2.0, to test sensitivity
        """
        template = torch.empty(1, seq_len, self.d_model, device=self.device, dtype=self.dtype)
        if kind == "zero":
            return torch.zeros_like(template)
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        state = self.model.initialize_state(template)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        return state * scale


    @torch.no_grad()
    def run(self, input_ids, num_steps, init_state=None, intervention=None,
            capture_blocks=(R4,)):
        """
        One forward pass. Returns the first-answer-token logits plus the captured block
        outputs for every loop (block_states[b][k-1] is block b's output on loop k).
        """
        core = self.model.transformer.core_block
        captured = {b: [] for b in capture_blocks}
        loop_counter = {"r4_calls": 0}

        def make_hook(block_idx):
            def hook(module, inputs, output):
                x = output[0]
                if block_idx == R4:
                    loop_counter["r4_calls"] += 1
                    k = loop_counter["r4_calls"]
                    if intervention is not None and intervention.loop == k:
                        x = x.clone()
                        pos = slice(None) if intervention.positions is None else intervention.positions
                        src = intervention.state.to(x.dtype)
                        if intervention.mode == "replace":
                            x[:, pos] = src[:, pos]
                        elif intervention.mode == "add":
                            x[:, pos] = x[:, pos] + src
                        else:
                            raise ValueError(intervention.mode)
                if block_idx in captured:
                    captured[block_idx].append(x.detach())
                if x is not output[0]:
                    return (x,) + tuple(output[1:])
                return None
            return hook

        hook_blocks = set(capture_blocks) | ({R4} if intervention is not None else set())
        handles = [core[b].register_forward_hook(make_hook(b)) for b in sorted(hook_blocks)]
        try:
            if init_state is None:
                init_state = self.make_init_state(input_ids.shape[1], kind="seed", seed=0)
            out = self.model(input_ids, input_states=init_state, num_steps=num_steps,
                             use_cache=False)
        finally:
            for h in handles:
                h.remove()

        if intervention is not None and loop_counter["r4_calls"] < intervention.loop:
            raise ValueError(f"asked to intervene at loop {intervention.loop} but only ran {loop_counter['r4_calls']} loops")

        logits = out.logits[0, -1].float()
        pred_id = int(logits.argmax())
        return RunOutput(logits=logits, pred_id=pred_id,
                         pred_text=self.tok.decode([pred_id]).strip(),
                         block_states=captured)


    @torch.no_grad()
    def logit_lens(self, state):
        """Unembed a hidden state directly: lm_head(ln_f(h)) at the last position."""
        h = self.model.transformer.ln_f(state[:, -1:])
        return self.model.lm_head(h)[0, -1].float()

    @torch.no_grad()
    def coda_lens(self, state):
        """Run the model's own 2 coda blocks on the state, then unembed (Lu et al. eq. 3)."""
        seq_len = state.shape[1]
        freqs_cis = self.model.freqs_cis[:, :seq_len]
        x = self.model.transformer.ln_f(state)
        for i, block in enumerate(self.model.transformer.coda, start=1):
            x, _ = block(x, freqs_cis, -i, None, None)
        x = self.model.transformer.ln_f(x)
        return self.model.lm_head(x)[0, -1].float()


    def digit_ids(self, digit):
        return self._digit_ids[str(digit)]

    @staticmethod
    def rank_of(logits, token_ids):
        """1-based rank of the best-ranked id in token_ids (1 = top prediction)."""
        best = max(logits[t] for t in token_ids)
        return int((logits > best).sum()) + 1

    def digit_rank(self, logits, digit):
        return self.rank_of(logits, self.digit_ids(digit))

    def digit_logit(self, logits, digit):
        return float(max(logits[t] for t in self.digit_ids(digit)))
