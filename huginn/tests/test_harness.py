"""Harness tests on a tiny random Huginn (CPU).  python -m tests.test_harness
"""

import json
import os

import torch

from huginn_harness import R4, Intervention, Runner, build_input_ids
from tests.tiny_huginn import load_tiny

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-4


def setup():
    model, tok = load_tiny()
    runner = Runner(model, tok)
    fewshot = json.load(open(os.path.join(ROOT, "data", "fewshot.json")))
    pair = json.load(open(os.path.join(ROOT, "data", "patch_pairs.json")))[0]
    src = build_input_ids(tok, fewshot, pair["source"], "cpu")
    tgt = build_input_ids(tok, fewshot, pair["target"], "cpu")
    return runner, src, tgt


def close(a, b):
    return torch.allclose(a, b, atol=TOL, rtol=0)


def test_same_seed_is_deterministic(runner, ids):
    init = runner.make_init_state(ids.shape[1], seed=3)
    a = runner.run(ids, 6, init).logits
    b = runner.run(ids, 6, runner.make_init_state(ids.shape[1], seed=3)).logits
    assert close(a, b)


def test_starting_state_variants_differ(runner, ids):
    L = ids.shape[1]
    s0, s1 = runner.make_init_state(L, seed=0), runner.make_init_state(L, seed=1)
    z = runner.make_init_state(L, kind="zero")
    half = runner.make_init_state(L, seed=0, scale=0.5)
    assert not close(s0, s1)
    assert z.abs().max() == 0
    assert close(half * 2, s0)
    assert runner.run(ids, 2, s0).logits.shape == runner.run(ids, 2, z).logits.shape


def test_init_does_not_touch_global_rng(runner, ids):
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    runner.make_init_state(ids.shape[1], seed=7)
    assert torch.equal(torch.rand(3), expected)


def test_captures_one_state_per_loop(runner, ids):
    out = runner.run(ids, 5, capture_blocks=(0, 1, 2, 3))
    assert all(len(out.block_states[b]) == 5 for b in range(4))
    assert out.block_states[R4][0].shape == (1, ids.shape[1], runner.d_model)


def test_coda_lens_at_loop_k_equals_running_k_loops(runner, ids):
    """exp3 relies on this: one R-loop run gives the model's answer at every loop count."""
    init = runner.make_init_state(ids.shape[1], seed=0)
    states = runner.run(ids, 8, init).block_states[R4]
    for k in (1, 3, 8):
        direct = runner.run(ids, k, init, capture_blocks=()).logits
        via_lens = runner.coda_lens(states[k - 1])
        assert close(direct, via_lens), f"loop {k}: max diff {(direct - via_lens).abs().max()}"


def test_self_patch_is_a_no_op(runner, ids):
    init = runner.make_init_state(ids.shape[1], seed=0)
    base = runner.run(ids, 6, init)
    for k in (1, 3, 6):
        patched = runner.run(ids, 6, init, capture_blocks=(),
                             intervention=Intervention(loop=k, state=base.block_states[R4][k - 1]))
        assert close(base.logits, patched.logits)


def test_patch_at_last_loop_gives_source_output(runner, src, tgt):
    R = 6
    s = runner.run(src, R, runner.make_init_state(src.shape[1], seed=0))
    t_init = runner.make_init_state(tgt.shape[1], seed=1)
    patched = runner.run(tgt, R, t_init, intervention=Intervention(loop=R, state=s.block_states[R4][-1]))
    assert close(patched.logits, s.logits)
    assert close(patched.block_states[R4][-1], s.block_states[R4][-1])


def test_patch_at_loop_k_equals_restarting_from_that_state(runner, src, tgt):
    """Swapping in A's state at loop k, then finishing on B's input, is the same as
    starting B from A's loop-k state and running the remaining R-k loops."""
    R, k = 8, 3
    s = runner.run(src, R, runner.make_init_state(src.shape[1], seed=0))
    t_init = runner.make_init_state(tgt.shape[1], seed=1)
    patched = runner.run(tgt, R, t_init, capture_blocks=(),
                         intervention=Intervention(loop=k, state=s.block_states[R4][k - 1]))
    restarted = runner.run(tgt, R - k, s.block_states[R4][k - 1], capture_blocks=())
    assert close(patched.logits, restarted.logits)


def test_patch_actually_changes_something(runner, src, tgt):
    R = 6
    s = runner.run(src, R, runner.make_init_state(src.shape[1], seed=0))
    t_init = runner.make_init_state(tgt.shape[1], seed=1)
    base = runner.run(tgt, R, t_init)
    patched = runner.run(tgt, R, t_init, intervention=Intervention(loop=2, state=s.block_states[R4][1]))
    assert not close(base.logits, patched.logits)


def test_last_position_only_patch(runner, src, tgt):
    R = 6
    s = runner.run(src, R, runner.make_init_state(src.shape[1], seed=0))
    t_init = runner.make_init_state(tgt.shape[1], seed=1)
    base = runner.run(tgt, R, t_init)
    patched = runner.run(tgt, R, t_init,
                         intervention=Intervention(loop=R, state=s.block_states[R4][-1], positions=[-1]))
    got = patched.block_states[R4][-1]
    assert close(got[:, -1], s.block_states[R4][-1][:, -1])
    assert close(got[:, :-1], base.block_states[R4][-1][:, :-1])


def test_zero_steering_vector_is_a_no_op(runner, ids):
    init = runner.make_init_state(ids.shape[1], seed=0)
    base = runner.run(ids, 5, init).logits
    steered = runner.run(ids, 5, init, capture_blocks=(),
                         intervention=Intervention(loop=2, state=torch.zeros(runner.d_model), mode="add")).logits
    assert close(base, steered)


def test_intervening_past_last_loop_raises(runner, ids):
    try:
        runner.run(ids, 3, intervention=Intervention(loop=5, state=torch.zeros(1)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_ranks_and_digits(runner, ids):
    logits = runner.run(ids, 3).logits
    top = int(logits.argmax())
    assert runner.rank_of(logits, [top]) == 1
    for d in range(10):
        assert runner.digit_ids(d), f"no token id for digit {d}"
        assert 1 <= runner.digit_rank(logits, d) <= logits.numel()
    assert runner.logit_lens(runner.run(ids, 2).block_states[R4][0]).shape == logits.shape


def test_prompt_matches_lu_et_al_format(runner, ids):
    text = runner.tok.decode(ids[0])
    assert "Always return only the final answer straightway." in text
    assert text.count("Question: What is") == 5          # 4 shots + the test question
    assert text.endswith("<|begin_header|>Huginn<|end_header|>\n\n")


def test_paired_prompts_have_equal_length(runner, src, tgt):
    assert src.shape == tgt.shape


def main():
    runner, src, tgt = setup()
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        args = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        kwargs = {"runner": runner, "ids": tgt, "src": src, "tgt": tgt}
        try:
            fn(*[kwargs[a] for a in args])
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
