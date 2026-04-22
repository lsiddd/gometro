"""
Verify that PyTorch CPU thread settings are correct and that the feature
extractor runs efficiently under training-like conditions.

Run with:
    uv run python test_torch_cpu.py
"""
from __future__ import annotations

import os
import time

# Apply the same env-var fix as train.py — must happen before import torch.
_N_CPU = str(os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS",      _N_CPU)
os.environ.setdefault("MKL_NUM_THREADS",      _N_CPU)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_CPU)

import torch
import numpy as np
import gymnasium as gym

torch.set_num_threads(int(_N_CPU))

from constants import OBS_DIM
from models import MetroFeatureExtractor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, reps: int = 50) -> float:
    """Return mean wall-clock seconds per call over `reps` repetitions."""
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_thread_count():
    got  = torch.get_num_threads()
    want = int(_N_CPU)
    assert got == want, f"thread count: want {want}, got {got}"
    print(f"  torch threads = {got}  ✓")


def test_blas_multithread_speedup():
    """Large matmul should be meaningfully faster with all cores than with 1."""
    N = 2048
    a = torch.randn(N, N)
    b = torch.randn(N, N)

    torch.set_num_threads(1)
    t_single = _time_fn(lambda: a @ b)

    torch.set_num_threads(int(_N_CPU))
    t_multi  = _time_fn(lambda: a @ b)

    speedup = t_single / t_multi
    gflops  = (2 * N**3) / t_multi / 1e9
    print(f"  matmul {N}×{N}: single={t_single*1e3:.1f} ms  "
          f"multi={t_multi*1e3:.1f} ms  speedup={speedup:.1f}×  "
          f"({gflops:.1f} GFLOPS)  ✓")
    # Require at least 1.5× speedup when more than 1 core available.
    if int(_N_CPU) > 1:
        assert speedup >= 1.5, (
            f"BLAS multi-thread speedup too low ({speedup:.1f}×) — "
            "OMP_NUM_THREADS may not have taken effect"
        )


def test_feature_extractor_forward():
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    model = MetroFeatureExtractor(obs_space)
    model.eval()

    batch = torch.randn(2048, OBS_DIM)  # training-size batch

    with torch.no_grad():
        out = model(batch)

    assert out.shape == (2048, 768), f"unexpected output shape {out.shape}"
    assert not torch.isnan(out).any(), "NaN in feature extractor output"

    ms = _time_fn(lambda: model(batch), reps=20) * 1e3
    print(f"  feature extractor forward: {ms:.1f} ms/call (batch=2048)  ✓")


def test_feature_extractor_backward():
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    model = MetroFeatureExtractor(obs_space)
    model.train()

    batch = torch.randn(2048, OBS_DIM)

    def _step():
        out = model(batch)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    ms = _time_fn(_step, reps=10) * 1e3
    print(f"  feature extractor forward+backward: {ms:.1f} ms/call (batch=2048)  ✓")


def test_compile_correctness():
    if not hasattr(torch, "compile"):
        print("  torch.compile not available — skipping  ✓")
        return

    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    batch = torch.randn(64, OBS_DIM)

    # Use same seed so both models start with identical weights.
    torch.manual_seed(0)
    eager = MetroFeatureExtractor(obs_space).eval()

    torch.manual_seed(0)
    to_compile = MetroFeatureExtractor(obs_space).eval()
    # Compile the module in-place (weights already loaded — no state_dict transfer needed).
    compiled = torch.compile(to_compile, mode="reduce-overhead")

    with torch.no_grad():
        out_eager    = eager(batch)
        out_compiled = compiled(batch)

    max_diff = (out_eager - out_compiled).abs().max().item()
    assert max_diff < 1e-4, f"compile output diverges from eager: max_diff={max_diff}"

    # Use training batch size to measure realistic speedup.
    train_batch = torch.randn(2048, OBS_DIM)
    ms_eager    = _time_fn(lambda: eager(train_batch),    reps=20) * 1e3
    ms_compiled = _time_fn(lambda: compiled(train_batch), reps=20) * 1e3
    speedup = ms_eager / ms_compiled
    print(f"  torch.compile (batch=2048): eager={ms_eager:.1f} ms  "
          f"compiled={ms_compiled:.1f} ms  speedup={speedup:.2f}×  ✓")


def test_update_cycle_estimate():
    """Estimate how long a full PPO update cycle takes vs collection time.

    Collection: n_steps=2048 steps × n_envs=12 envs at ~Go simulation speed.
    Update:     n_epochs=4 × ceil(24576 / batch_size=2048) = 48 grad steps.
    """
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    model = MetroFeatureExtractor(obs_space)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch = torch.randn(2048, OBS_DIM)

    def _grad_step():
        opt.zero_grad()
        out = model(batch)
        loss = out.sum()
        loss.backward()
        opt.step()

    ms_per_step = _time_fn(_grad_step, reps=10) * 1e3
    n_grad_steps = 48  # 4 epochs × 12 minibatches
    total_update_s = ms_per_step * n_grad_steps / 1000

    # Rough estimate: Go sim runs ~1000 steps/s/env (conservative)
    n_envs, n_steps = 12, 2048
    collection_s = n_steps / 1000  # 1000 steps/s per env (Go is faster but conservative)

    ratio = total_update_s / collection_s
    print(f"  grad step: {ms_per_step:.1f} ms  ×  {n_grad_steps} steps  "
          f"= {total_update_s:.1f}s update  vs  ~{collection_s:.1f}s collection  "
          f"(update/collect ratio = {ratio:.1f}×)  ✓")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"CPU count: {_N_CPU}  |  torch {torch.__version__}\n")

    tests = [
        test_thread_count,
        test_blas_multithread_speedup,
        test_feature_extractor_forward,
        test_feature_extractor_backward,
        test_compile_correctness,
        test_update_cycle_estimate,
    ]

    failed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"[{name}]")
            t()
        except AssertionError as exc:
            print(f"  FAIL: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    print(f"\n{'All tests passed' if not failed else f'{failed} test(s) failed'}")
    raise SystemExit(failed)
