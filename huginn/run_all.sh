#!/usr/bin/env bash
# Runs everything in order on one GPU. Rough estimate: 30+ minutes on an L4, about twice that on a T4.
# Each step prints a short summary; plots and JSON land in results/.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 0. sanity tests on a tiny random Huginn (CPU, ~10s)"
python -m tests.test_harness

echo "== 1a. accuracy on the 50 problems (4 / 8 / 16 / 32 loops)"
python exp1_accuracy.py

echo "== 1b. accuracy on the 300-problem pool, so exp2 has enough correct problems"
python exp1_accuracy.py --data two_step_pool.json --out-prefix pool

echo "== 2. replicate Lu et al. Fig 4 (rank trajectories) on problems Huginn gets right"
python exp2_rank_trajectory.py --data results/pool_correct_r16.json

echo "== 3. random starting state: does the initial noise change the answer, and when do starts converge?"
python exp3_random_start.py

echo "== 4. first hybrid-answer patching test"
python exp4_patch_smoke.py
python exp4_patch_smoke.py --positions last

echo "done. see results/*.png"
