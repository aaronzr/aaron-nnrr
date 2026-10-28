# Coda lens replication of Lu et al. (2025)

Model huginn-0125 @ 2a364bd, 16 loops, seed 42, device cuda

Exact equality with the paper isn't expected (GPU kernels and library versions differ, and bf16 ties can reorder very low-ranked tokens). The shapes are what should match.

## Fig 4 (coda lens at R4) and Appendix Fig 6 (R1-R3)

- Huginn answered **66/67** of these correctly in our run (the paper picked them because it got all of them right; a few misses are run-to-run noise).

| Block | Token | Ours: mean rank loop 1 / 6 / 16 | Paper: loop 1 / 6 / 16 | Curve shape (Spearman) |
| --- | --- | --- | --- | --- |
| R1 | final | 3816 / 1765 / 539 | 792 / 1101 / 535 | 0.85 |
| R1 | intermediate | 5801 / 2316 / 1932 | 2310 / 1632 / 1936 | 0.76 |
| R1 | 'the' | 8396 / 60121 / 53177 | 1231 / 58456 / 53324 | 0.94 |
| R2 | final | 163 / 1755 / 421 | 141 / 1639 / 398 | 0.95 |
| R2 | intermediate | 183 / 1744 / 1341 | 207 / 1661 / 1333 | 0.86 |
| R2 | 'the' | 2222 / 56543 / 51455 | 457 / 55949 / 51805 | 0.94 |
| R3 | final | 87 / 22 / 4 | 72 / 16 / 4 | 0.98 |
| R3 | intermediate | 101 / 50 / 14 | 104 / 31 / 13 | 0.96 |
| R3 | 'the' | 856 / 20104 / 14260 | 280 / 19018 / 14427 | 0.98 |
| R4 | final | 65 / 4 / 1 | 316 / 4 / 1 | 1.00 |
| R4 | intermediate | 164 / 4 / 4 | 462 / 3 / 4 | 0.95 |
| R4 | 'the' | 2755 / 330 / 257 | 896 / 696 / 273 | 0.87 |

- Final token ranked better than the intermediate (mean, at R4) on **81%** of loops (paper: 'the final token consistently maintaining a lower rank').
- Share of questions where the intermediate outranks the final, by loop at R4 (the 'rank reversal'):
  - ours:  0.31 0.34 0.34 0.30 0.54 0.49 0.69 0.64 0.48 0.10 0.06 0.01 0.06 0.04 0.01 0.00
  - paper: 0.45 0.31 0.30 0.30 0.61 0.63 0.69 0.57 0.40 0.10 0.04 0.03 0.06 0.09 0.01 0.00
  - peak: loop 7 for ours, loop 7 for the paper


Total time: 1195s