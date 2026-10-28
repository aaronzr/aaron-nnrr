# Huginn experiments

Code for probing and intervening on Huginn-0125's recurrent loop
(`tomg-group-umd/huginn-0125`, revision `2a364bd`, same as Lu et al.). Inference only.

```bash
pip install -r requirements.txt
python -m tests.test_harness      # CPU, tiny random model
```

## Layout

| Path | What |
| --- | --- |
| `huginn_harness.py` | model loading, prompts, per-loop state capture, R4 patching/steering, logit and coda lens |
| `lu_replication/` | coda lens replication of Lu et al. (2025), with results in `lu_replication/results_coda/` |
| `reproduce_geiping/` | lm-eval scripts for the Geiping et al. (2025) benchmark tables |
| `make_dataset.py`, `data/` | two-step arithmetic set (50 + 300 pool), few-shot examples, patch pairs |
| `exp1_accuracy.py` | accuracy vs loops on the two-step set |
| `exp2_rank_trajectory.py` | final / intermediate / "the" rank per block per loop, both lenses |
| `exp3_random_start.py` | effect of the random starting state (seeds, zeros, scaled noise) |
| `exp4_patch_smoke.py` | hybrid-answer patching at R4 with self-patch and noise controls |
| `tests/` | tiny 64-dim Huginn + harness tests |

## Notes

- Every run passes an explicit starting state. Without one, Huginn draws new noise on each forward pass.
- Answers are single tokens, so each question is one forward pass with `use_cache=False` (about 7 GB on a T4). `generate()` keeps a KV cache per loop and runs out of memory quickly.
- The coda lens at R4 after loop k equals the model's output with `num_steps=k` (tested), so one run gives the answer at every loop count.
- `transformers` must be <5.
- Huginn's current HF code ignores the attention mask, so batched generation with left padding is wrong. Generation runs at batch size 1.

## Colab

`lu_replication/coda_lens_colab.ipynb` clones this branch and runs the replication on a T4.
