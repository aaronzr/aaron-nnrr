"""Offline test for compare_results.py.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_results as cr  # noqa: E402

MODEL = "tomg-group-umd__huginn-0125"


def write(root, name, loops, results, n_samples, sys_prompt=None, model_args_as_dict=False):
    args = f"pretrained=tomg-group-umd/huginn-0125,trust_remote_code=True,dtype=bfloat16,mean_recurrence={loops}"
    if model_args_as_dict:
        args = dict(p.split("=", 1) for p in args.split(","))
    d = Path(root) / name / MODEL
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_2026-10-04T13-00-00.json").write_text(json.dumps({
        "results": results, "n-samples": n_samples,
        "config": {"model": "hf", "model_args": args, "batch_size": "auto"},
        "system_instruction": sys_prompt,
    }))


def mc(acc_norm, se=0.009, acc=None):
    return {"alias": "x", "acc,none": acc if acc is not None else acc_norm, "acc_stderr,none": se,
            "acc_norm,none": acc_norm, "acc_norm_stderr,none": se}


def main():
    with tempfile.TemporaryDirectory() as root:
        # full ARC-E at 32 loops, right on the paper (69.91)
        write(root, "table1_r32", 32, {"arc_easy": mc(0.6995), "winogrande": mc(0.5, acc=0.5943)},
              {"arc_easy": {"original": 2376, "effective": 2376}, "winogrande": {"original": 1267, "effective": 1267}})
        # ARC-E at 4 loops, off by 10 points; model_args saved as a dict
        write(root, "table1_r4", 4, {"arc_easy": mc(0.39)},
              {"arc_easy": {"original": 2376, "effective": 2376}}, model_args_as_dict=True)
        # partial run (--limit 100)
        write(root, "smoke_arc_r8", 8, {"arc_easy": mc(0.66, se=0.047)},
              {"arc_easy": {"original": 2376, "effective": 100}})
        # GSM8K CoT with system prompt: strict 34.0 (close to 34.80), flexible 42.0 (paper 42.08)
        write(root, "gsm8k_r32", 32, {"gsm8k_cot": {
            "exact_match,strict-match": 0.340, "exact_match_stderr,strict-match": 0.0335,
            "exact_match,flexible-extract": 0.420, "exact_match_stderr,flexible-extract": 0.035}},
            {"gsm8k_cot": {"original": 1319, "effective": 200}},
            sys_prompt=cr.TARGETS["system_instruction"])
        # GSM8K CoT without system prompt (paper 32.60/34.57)
        write(root, "gsm8k_nosys_r32", 32, {"gsm8k_cot": {
            "exact_match,strict-match": 0.20, "exact_match_stderr,strict-match": 0.01,
            "exact_match,flexible-extract": 0.346, "exact_match_stderr,flexible-extract": 0.013}},
            {"gsm8k_cot": {"original": 1319, "effective": 1319}})
        # sanity at 1 loop: paper 0.00
        write(root, "gsm8k_sanity_r1", 1, {"gsm8k_cot": {
            "exact_match,strict-match": 0.0, "exact_match_stderr,strict-match": 0.0,
            "exact_match,flexible-extract": 0.0, "exact_match_stderr,flexible-extract": 0.0}},
            {"gsm8k_cot": {"original": 1319, "effective": 50}}, sys_prompt="x")
        # code
        write(root, "code_r32", 32, {"humaneval_instruct": {"pass@1,create_test": 0.2317,
                                                            "pass@1_stderr,create_test": 0.033}},
              {"humaneval_instruct": {"original": 164, "effective": 164}})
        # a corrupt file must be skipped, not crash
        bad = Path(root) / "broken" / MODEL
        bad.mkdir(parents=True)
        (bad / "results_bad.json").write_text("{not json")

        runs = cr.load_runs(root)
        assert len(runs) == 7, len(runs)
        rows = {(r[0], r[1], r[2]): r for run in runs for r in cr.rows_for_run(run)}

        def check(key, paper, verdict_prefix):
            row = rows[key]
            assert row[6] == paper, (key, row[6], paper)
            assert row[8].startswith(verdict_prefix), (key, row[8], verdict_prefix)

        check(("arc_easy", "acc_norm", 32), 69.91, "MATCH")
        check(("winogrande", "acc", 32), 59.43, "MATCH")          # winogrande uses acc, not acc_norm
        check(("arc_easy", "acc_norm", 4), 49.07, "OFF")
        assert rows[("arc_easy", "acc_norm", 8)][8] == "MATCH (PARTIAL)", rows[("arc_easy", "acc_norm", 8)]
        check(("gsm8k_cot (sys prompt)", "strict-match", 32), 34.80, "MATCH")
        check(("gsm8k_cot (sys prompt)", "flexible-extract", 32), 42.08, "MATCH")
        check(("gsm8k_cot (no sys prompt)", "strict-match", 32), 32.60, "OFF")
        check(("gsm8k_cot (no sys prompt)", "flexible-extract", 32), 34.57, "MATCH")
        check(("gsm8k_cot (sys prompt)", "strict-match", 1), 0.0, "MATCH")
        check(("humaneval_instruct", "pass@1", 32), 23.17, "MATCH")

        # the CLI runs end to end and writes a CSV
        csv_path = Path(root) / "cmp.csv"
        out = subprocess.run([sys.executable, str(HERE / "compare_results.py"), root, "--csv", str(csv_path)],
                             capture_output=True, text=True, check=True).stdout
        assert "arc_easy" in out and "r4: 39.0 -> r8: 66.0 -> r32: 70.0" in out, out
        assert csv_path.exists() and len(csv_path.read_text().splitlines()) == len(rows) + 1
        print(out)
    print("all compare_results checks passed")


if __name__ == "__main__":
    main()
