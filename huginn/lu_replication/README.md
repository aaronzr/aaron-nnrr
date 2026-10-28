# Coda lens replication: Lu et al. (2025)

"Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer" (arXiv 2507.02199).
Their code: https://github.com/wenquanlu/huginn-latent-cot

```bash
python lu_replication/coda_lens_replication.py --part fig23   # ~31 min on a T4
python lu_replication/coda_lens_replication.py --part fig4    # ~20 min on a T4
python lu_replication/coda_lens_replication.py --limit 5 --out results_check
```

Or run `coda_lens_colab.ipynb` in Colab.

## Setup (matched to their repo)

- huginn-0125 at revision `2a364bd`, bf16, 16 loops
- `set_seed(42)` before every question (fixes the random starting state)
- their system prompt and 4-shot chat format, first answer token at the last position
- Fig 2/3: first 104 items of GPT-3 `single_digit_three_ops` (= `arithmetic_1dc`), items 0-3 few-shot, 4-103 test
- Fig 4/6: their `filtered_arithmetic_dataset.json` (4 few-shot + 67)
- coda lens: `RMSNorm -> 2 coda blocks -> RMSNorm -> lm_head`
- numeric tokens: their regex `(-\d+|\d+|-)`

`data/lu_reference/` holds their saved pickles (coda ranks and top-5 tokens), which every
result is compared against.

## Results (T4, transformers 4.57)

| Check | Ours | Paper |
| --- | --- | --- |
| predicted first token matches their list | 97/100 | |
| Fig 2 rank curve, Spearman over 66 states | 0.987 | |
| mean rank at R4 / R1-R3 | 20.2 / 3137 | 4.2 / 3079 |
| rank at final state (R4, loop 16) | 1.00 | 1.00 |
| Fig 3 numeric share at loop 16, R3 / R4 | 0.43 / 0.99 | 0.46 / 0.99 |
| Fig 4 accuracy on the 67 filtered questions | 66/67 | 67/67 |
| Fig 4 R4 final-token curve, Spearman | 1.00 | |
| rank reversal peak (intermediate beats final) | loop 7, 69% | loop 7, 69% |

The R4 discontinuity, the numeric content at R4, and the lack of an intermediate-then-final
ordering all replicate. Loops 1-2 differ the most (the state is still mostly the random
starting noise there). From loop 4 on the curves overlap.

Full numbers are in `results_coda/summary_fig23.md` and `results_coda/summary_fig4.md`.

## Outputs (`results_coda/`)

| File | Shape / content |
| --- | --- |
| `fig2_coda_ranks.npy` | [100, 66] rank of the predicted token (2 prelude + 64 core states) |
| `fig2_fig3_coda_details.json` | predicted token ids, top-5 tokens per core state |
| `fig4_coda_{correct,inter,the}_ranks.npy` | [67, 64] ranks of final / intermediate / "the" |
| `fig2_fig3_coda.png`, `fig4_fig6_coda.png` | plots, ours solid, paper dashed |
| `summary_*.md` | comparison tables |
