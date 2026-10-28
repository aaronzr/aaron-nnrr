# Coda lens replication of Lu et al. (2025)

Model huginn-0125 @ 2a364bd, 16 loops, seed 42, device cuda

Exact equality with the paper isn't expected (GPU kernels and library versions differ, and bf16 ties can reorder very low-ranked tokens). The shapes are what should match.

## Fig 2 (coda lens, all 66 states)

- Predicted first token matches the paper's own list on **97/100** questions (checks that prompt, seed and model match theirs).
- Shape of the mean-rank curve vs the paper: Spearman correlation **0.987** across the 66 positions.
- R4 dips (paper: ranks drop sharply at R4): mean rank at R4 = **20.2** (paper 4.2); at R1-R3 = **3137** (paper 3079).
- Final state (R4, loop 16): mean rank 1.00 (paper 1.00); should be about 1, since the coda lens at the last R4 is the model's real output.

## Fig 3 (coda lens, share of top-5 tokens that are numbers or '-')

| Block | Ours (mean over loops) | Paper | Ours at loop 16 | Paper at loop 16 |
| --- | --- | --- | --- | --- |
| R1 | 0.00 | 0.00 | 0.00 | 0.00 |
| R2 | 0.01 | 0.01 | 0.00 | 0.00 |
| R3 | 0.36 | 0.37 | 0.43 | 0.46 |
| R4 | 0.89 | 0.92 | 0.99 | 0.99 |

Largest gap at any block and loop: 0.24

## Appendix Tables 5-7 (coda lens): top-5 tokens, first 3 questions

**Loop 1**

| Block | Ours | Paper |
| --- | --- | --- |
| R1 | [' Answer', 'Answer', ' personality', ' hum', ' soft']<br>[' Answer', ' personality', 'Answer', ' hum', ' soft']<br>[' Answer', ' personality', ' hum', 'Answer', ' soft'] | [' plasma', ' sounds', ' draft', '```', 'Answer']<br>[' plasma', ' sounds', ' draft', '```', ' functions']<br>[' plasma', ' sounds', ' draft', ' functions', '```'] |
| R2 | ['Answer', '**', '-', 'Given', ' Answer']<br>['Answer', '**', '-', 'Given', ' Answer']<br>['Answer', '**', '-', 'Given', ' Answer'] | ['Answer', '-', '**', '```', 'You']<br>['Answer', '-', '**', '```', ' plasma']<br>['Answer', '-', '**', ' plasma', '```'] |
| R3 | ['Answer', '-', '**', 'Question', '```']<br>['Answer', '-', '**', 'Question', '```']<br>['Answer', '-', '**', '```', 'Question'] | ['```', 'Answer', '-', '**', '####']<br>['Answer', '```', '-', '**', 'You']<br>['Answer', '```', '-', '**', '####'] |
| R4 | ['Question', '-', 'The', 'Answer', 'What']<br>['Question', '-', 'The', 'Answer', 'What']<br>['Question', '-', 'The', 'Answer', 'What'] | ['-', 'Question', 'Your', 'Answer', 'The']<br>['-', 'Question', 'Your', 'Answer', 'The']<br>['-', 'Question', 'Your', 'Answer', 'The'] |

**Loop 8**

| Block | Ours | Paper |
| --- | --- | --- |
| R1 | [' inc', ' Answer', 'Answer', ' answer', 'IF']<br>[' answering', ' answer', ' Answer', ' templates', ' bul']<br>[' answer', ' answering', ' accurate', ' spa', ' Answers'] | [' inc', ' answer', ' Answer', 'Answer', ' answering']<br>[' answer', ' answering', ' Answer', ' bul', ' templates']<br>[' answer', ' answering', ' spa', ' Answers', ' answers'] |
| R2 | [' inc', '\\boxed', ' answer', ' Answer', ' \\boxed']<br>[' answer', ' answering', ' Answer', ' tub', ' answers']<br>[' answer', ' answering', ' answers', ' Answers', ' greeting'] | [' answer', ' inc', '\\boxed', ' Answer', ' \\boxed']<br>[' answer', ' answering', ' Answer', ' answers', ' tub']<br>[' answer', ' answering', ' answers', ' Answers', ' greeting'] |
| R3 | ['<|end_turn|>', '\\boxed', '4', '1', 'Answer']<br>['3', '4', '2', '8', '5']<br>['Answer', ' answer', '8', ' answering', '\\boxed'] | ['<|end_turn|>', '\\boxed', '4', '2', '1']<br>['3', '\\boxed', '2', '4', '8']<br>['Answer', '<|end_turn|>', ' answer', '3', '8'] |
| R4 | ['1', '2', '4', '9', '8']<br>['4', '2', '3', '1', '8']<br>['1', '4', '2', '8', '3'] | ['1', '2', '9', '4', '8']<br>['4', '2', '3', '1', '8']<br>['1', '4', '2', '8', '6'] |

**Loop 16**

| Block | Ours | Paper |
| --- | --- | --- |
| R1 | [' answer', ' swimming', ' tweets', ' cour', ' inc']<br>[' answering', ' bul', ' spa', ' answer', ' ships']<br>[' answering', ' answer', ' optics', ' spa', ' tweets'] | [' answer', ' swimming', ' tweets', ' cour', ' inc']<br>[' answering', ' bul', ' answer', ' spa', ' Answer']<br>[' answering', ' answer', ' optics', ' spa', ' tweets'] |
| R2 | [' answer', ' Answer', ' wall', ' responses', 'Answer']<br>[' answer', ' answering', ' Answer', ' answers', ' tanks']<br>[' answer', ' answering', ' Answer', ' radicals', ' answers'] | [' answer', ' Answer', ' wall', ' responses', ' decisions']<br>[' answer', ' answering', ' Answer', ' answers', ' tanks']<br>[' answer', ' answering', ' Answer', ' radicals', ' answers'] |
| R3 | ['1', 'Answer', '4', '\\boxed', '8']<br>['Answer', ' Answer', '4', '3', ' answer']<br>['8', 'Answer', '4', '9', '1'] | ['1', 'Answer', '4', '\\boxed', '8']<br>['Answer', ' Answer', '4', '3', ' answer']<br>['8', 'Answer', '4', '9', '1'] |
| R4 | ['1', '3', '2', '-', '4']<br>['4', '3', '2', '1', '8']<br>['1', '-', '8', '2', '6'] | ['1', '3', '2', '-', '4']<br>['4', '3', '2', '1', '8']<br>['1', '8', '-', '2', '6'] |


Total time: 1865s