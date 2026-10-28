#!/usr/bin/env bash
# Geiping et al. (2025) Huginn-0125 benchmarks via lm-eval.
# Usage: bash run_repro.sh <tier> [<tier> ...]
#   smoke | table1-fast | table1 | table1-big | math-sanity | math | code
# Env: OUT (default outputs/repro), LIMIT (N or none), MODEL
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-tomg-group-umd/huginn-0125}"
OUT="${OUT:-outputs/repro}"
SYS_PROMPT="You are a helpful assistant that can assist users with mathematical reasoning."
TABLE1_SMALL="arc_easy,arc_challenge,openbookqa,piqa,sciq,winogrande"

COMMON=(--seed 0)

model_args() {  # $1 = number of loops
  echo "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,mean_recurrence=$1"
}

limit_flag() {  # $1 = default cap, or "none"
  local cap="${LIMIT:-$1}"
  if [[ "$cap" == "none" ]]; then echo ""; else echo "--limit $cap"; fi
}

run_loglik() {  # $1 = loops, $2 = tasks, $3 = default cap, $4 = output name
  echo ">> $2 at $1 loops (cap: ${LIMIT:-$3})"
  lm_eval --model hf --model_args "$(model_args "$1")" \
    --tasks "$2" --num_fewshot 0 --batch_size auto \
    $(limit_flag "$3") "${COMMON[@]}" \
    --output_path "${OUT}/$4_r$1"
}

run_gsm8k() {  # $1 = loops, $2 = default cap, $3 = output name
  echo ">> GSM8K CoT with system prompt at $1 loops (cap: ${LIMIT:-$2})"
  # batch 1: HF code ignores the attention mask and generation left-pads
  lm_eval --model hf --model_args "$(model_args "$1")" \
    --tasks gsm8k_cot --batch_size 1 \
    --apply_chat_template --fewshot_as_multiturn \
    --system_instruction "${SYS_PROMPT}" \
    $(limit_flag "$2") "${COMMON[@]}" --log_samples \
    --output_path "${OUT}/$3_r$1"
}

compare() { python compare_results.py "${OUT}"; }

if [[ $# -eq 0 ]]; then
  sed -n '2,5p' "$0"; exit 1
fi

for tier in "$@"; do
  case "$tier" in
    smoke)
      run_loglik 4  arc_easy 100 smoke_arc
      run_loglik 32 arc_easy 100 smoke_arc
      run_gsm8k 32 3 smoke_gsm8k
      ;;
    table1-fast)
      for r in 4 8 16 32; do run_loglik "$r" "$TABLE1_SMALL" 500 table1fast; done
      ;;
    table1)
      for r in 4 8 16 32; do run_loglik "$r" "$TABLE1_SMALL" none table1; done
      ;;
    table1-big)
      run_loglik 32 hellaswag,mmlu 1000 table1big
      ;;
    math-sanity)
      run_gsm8k 1 50 gsm8k_sanity
      ;;
    math)
      run_gsm8k 32 100 gsm8k
      ;;
    code)
      echo ">> HumanEval (instruct) at 32 loops. Executes model-generated code."
      HF_ALLOW_CODE_EVAL=1 lm_eval --model hf --model_args "$(model_args 32)" \
        --tasks humaneval_instruct --batch_size 1 --num_fewshot 0 \
        --apply_chat_template --confirm_run_unsafe_code \
        --gen_kwargs do_sample=True,temperature=0.2,top_p=0.95 \
        $(limit_flag none) "${COMMON[@]}" --log_samples \
        --output_path "${OUT}/code_r32"
      ;;
    *)
      echo "unknown tier: $tier"; exit 1
      ;;
  esac
  compare
done
