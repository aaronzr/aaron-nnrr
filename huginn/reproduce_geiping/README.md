# Geiping et al. (2025) benchmark reproduction

Evaluates the released `tomg-group-umd/huginn-0125` with lm-eval at different loop counts
(`mean_recurrence`), following the commands in https://github.com/seal-rg/recurrent-pretraining.
No training.

```bash
pip install "lm-eval[hf]" "transformers<5"
bash run_repro.sh smoke
bash run_repro.sh math-sanity table1-fast math
python compare_results.py outputs/repro --csv comparison.csv
```

## Tiers

| Tier | What |
| --- | --- |
| `smoke` | ARC-E (100 qs) at 4 and 32 loops + 3 GSM8K qs |
| `math-sanity` | GSM8K CoT at 1 loop, 50 qs (paper: 0.00) |
| `table1-fast` | ARC-E, ARC-C, OBQA, PiQA, SciQ, WinoGrande at 4/8/16/32 loops, max 500 qs each |
| `table1` | same, full test sets |
| `math` | GSM8K CoT with system prompt at 32 loops, 100 qs |
| `table1-big` | HellaSwag + MMLU at 32 loops, max 1000 qs |
| `code` | HumanEval (instruct) at 32 loops |

`LIMIT=N` (or `LIMIT=none`) overrides the question cap.

## Targets (`paper_targets.json`)

Table 1, zero-shot:

| Loops | ARC-E | ARC-C | HellaSwag | MMLU | OBQA | PiQA | SciQ | WinoGrande |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 49.07 | 27.99 | 43.46 | 23.39 | 28.20 | 64.96 | 80.00 | 55.24 |
| 8 | 65.11 | 35.15 | 58.54 | 25.29 | 35.40 | 73.45 | 92.10 | 55.64 |
| 16 | 69.49 | 37.71 | 64.67 | 31.25 | 37.60 | 75.79 | 93.90 | 57.77 |
| 32 | 69.91 | 38.23 | 65.21 | 31.38 | 38.80 | 76.22 | 93.50 | 59.43 |

Table 2, GSM8K at 32 loops (strict / flexible): with system prompt 24.87 / 38.13 (CoT 34.80 / 42.08),
without 28.05 / 28.20 (CoT 32.60 / 34.57). Table 4: GSM8K CoT at 1 loop is 0.00.

## Notes

- Generation runs at batch size 1: Huginn's HF code ignores the attention mask and generation pads on the left.
- `--seed 0` fixes the starting noise; small differences from the paper are expected.
- lm-eval loads the latest Huginn code from HF (same weights as the paper).
