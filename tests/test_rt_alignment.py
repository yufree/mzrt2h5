"""
RT alignment quality test using simulated data with known RT shifts.

Strategy
--------
1. Simulate N samples from identical compounds; each sample gets a distinct
   global RT offset drawn from U(-max_shift_s, +max_shift_s).
2. Convert all mzML files to a single H5 file.
3. Measure mean pairwise BPC Pearson correlation *before* alignment.
4. Apply align_rt() using the reference sample (shift = 0) as the QC anchor.
5. Re-measure BPC correlation *after* alignment.
6. Report the improvement and per-sample estimated vs. true shift.

Run
---
    python test_rt_alignment.py [--samples N] [--compounds N]
                                [--max-shift S] [--seed SEED]
"""

import argparse
import os
import shutil
import tempfile

import h5py
import numpy as np
import pandas as pd

from mzrt2h5.alignment import align_rt, _build_all_bpc_streaming
from mzrt2h5.processing import save_dataset_as_sparse_h5
from mzrt2h5.simulation import generate_simulation_data


# ── helpers ──────────────────────────────────────────────────────────────────

def _generate_shifted_samples(n_samples, n_compounds, max_shift_s,
                               rtrange, seed, work_dir):
    """Generate N mzML files with known global RT shifts.

    All samples share the same compounds and reference RT positions.
    Sample 0 has shift = 0 (acts as the alignment reference).

    Returns
    -------
    mzml_dir : str   flat directory holding all mzML files
    meta_csv : str   metadata CSV (columns: Sample Name, true_shift_s)
    true_shifts : ndarray  (n_samples,) true RT shifts in seconds
    """
    rng = np.random.default_rng(seed)

    # Reference RT positions, shared by all samples
    ref_rtimes = np.sort(rng.uniform(*rtrange, size=n_compounds))

    # True shifts: sample 0 is the unshifted reference
    true_shifts = np.concatenate([
        [0.0],
        rng.uniform(-max_shift_s, max_shift_s, n_samples - 1),
    ])

    mzml_dir = os.path.join(work_dir, "mzml")
    os.makedirs(mzml_dir)

    meta_rows = []
    for i, shift in enumerate(true_shifts):
        rtimes = np.clip(ref_rtimes + shift, *rtrange).tolist()
        result = generate_simulation_data(
            n_compounds=n_compounds,
            rtrange=rtrange,
            rtime=rtimes,
            pwidth=8,
            snr=200,
            noise_sd=0.3,
            seed=seed,          # same seed → same compounds, reproducible
            output_dir=os.path.join(work_dir, f"sim_{i:02d}"),
        )
        src = result["mzml_path"]
        dst = os.path.join(mzml_dir, os.path.basename(src))
        shutil.copy(src, dst)
        meta_rows.append({
            "Sample Name": os.path.basename(dst).replace(".mzML", ""),
            "true_shift_s": round(float(shift), 3),
        })

    meta_df = pd.DataFrame(meta_rows)
    meta_csv = os.path.join(work_dir, "meta.csv")
    meta_df.to_csv(meta_csv, index=False)

    return mzml_dir, meta_csv, true_shifts


def _mean_pairwise_corr(h5_path, bin_size_s=1.0):
    """Mean pairwise Pearson correlation of BPC traces across all samples."""
    all_bpc, *_ = _build_all_bpc_streaming(h5_path, bin_size_s=bin_size_s)
    n = all_bpc.shape[0]
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = all_bpc[i], all_bpc[j]
            if a.std() == 0 or b.std() == 0:
                continue
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else float("nan")


def _estimated_shifts(h5_path, ref_name, bin_size_s=1.0):
    """Estimate per-sample RT shift relative to the reference via BPC argmax lag.

    Returns a dict {sample_name: estimated_shift_s}.
    """
    all_bpc, storage_rt_prec, n_bins, _, sample_names = \
        _build_all_bpc_streaming(h5_path, bin_size_s=bin_size_s)

    ref_idx = sample_names.index(ref_name) if ref_name in sample_names else 0
    ref_bpc = all_bpc[ref_idx]

    estimates = {}
    for i, name in enumerate(sample_names):
        if i == ref_idx:
            estimates[name] = 0.0
            continue
        # Cross-correlate BPCs to find lag
        n = min(len(ref_bpc), len(all_bpc[i]))
        xcorr = np.correlate(
            ref_bpc[:n] - ref_bpc[:n].mean(),
            all_bpc[i][:n] - all_bpc[i][:n].mean(),
            mode="full",
        )
        lag_bins = np.argmax(xcorr) - (n - 1)
        estimates[name] = float(lag_bins * bin_size_s)
    return estimates, sample_names


# ── main test ─────────────────────────────────────────────────────────────────

def run_test(n_samples=8, n_compounds=60, max_shift_s=30.0, seed=42):
    work_dir = tempfile.mkdtemp(prefix="rt_align_test_")
    h5_path  = os.path.join(work_dir, "dataset.h5")

    try:
        print(f"\n{'='*60}")
        print(f"RT alignment test")
        print(f"  samples={n_samples}  compounds={n_compounds}  "
              f"max_shift=±{max_shift_s}s  seed={seed}")
        print(f"{'='*60}\n")

        # ── 1. generate ───────────────────────────────────────────────
        print("Step 1: generating shifted mzML samples...")
        mzml_dir, meta_csv, true_shifts = _generate_shifted_samples(
            n_samples, n_compounds, max_shift_s,
            rtrange=(60, 540), seed=seed, work_dir=work_dir,
        )
        print(f"  True shifts (s): {np.round(true_shifts, 1).tolist()}\n")

        # ── 2. convert to H5 ──────────────────────────────────────────
        print("Step 2: converting mzML → H5...")
        save_dataset_as_sparse_h5(
            folder=mzml_dir,
            save_path=h5_path,
            rt_precision=0.1,
            mz_precision=0.01,
            metadata_csv_path=meta_csv,
        )
        print()

        # ── 3. correlation before alignment ───────────────────────────
        print("Step 3: BPC correlation BEFORE alignment...")
        corr_before = _mean_pairwise_corr(h5_path)
        print(f"  mean pairwise BPC r = {corr_before:.4f}\n")

        # identify reference sample (shift=0, first in list)
        with h5py.File(h5_path, "r") as f:
            sample_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in f["sample_id"][:]
            ]
        ref_name = sample_names[0]

        est_before, _ = _estimated_shifts(h5_path, ref_name)

        # ── 4. align ──────────────────────────────────────────────────
        print(f"Step 4: running align_rt() (reference: '{ref_name}')...")
        align_rt(
            h5_path,
            qc_sample_names=[ref_name],
            max_shift_s=max_shift_s + 10,   # slightly wider than true range
        )
        print()

        # ── 5. correlation after alignment ────────────────────────────
        print("Step 5: BPC correlation AFTER alignment...")
        corr_after = _mean_pairwise_corr(h5_path)
        print(f"  mean pairwise BPC r = {corr_after:.4f}\n")

        est_after, _ = _estimated_shifts(h5_path, ref_name)

        # ── 6. report ─────────────────────────────────────────────────
        print(f"{'─'*60}")
        print(f"{'Sample':<30}  {'True Δ':>8}  {'Est Δ before':>12}  "
              f"{'Est Δ after':>11}  {'Residual':>9}")
        print(f"  (residual = |Est Δ after| — how much offset remains)")
        print(f"{'─'*60}")
        residuals = []
        for i, name in enumerate(sample_names):
            true = true_shifts[i] if i < len(true_shifts) else float("nan")
            eb   = est_before.get(name, float("nan"))
            ea   = est_after.get(name, float("nan"))
            res  = abs(ea) if not np.isnan(ea) else float("nan")
            residuals.append(res)
            print(f"  {name:<28}  {true:>+8.1f}  {eb:>+12.1f}  "
                  f"{ea:>+11.1f}  {res:>9.1f}")
        print(f"{'─'*60}")

        improvement = corr_after - corr_before
        good = improvement > 0
        print(f"\nSummary")
        print(f"  BPC correlation:  {corr_before:.4f}  →  {corr_after:.4f}  "
              f"({'+' if improvement >= 0 else ''}{improvement:.4f})")
        valid_res = [r for r in residuals if not np.isnan(r)]
        if valid_res:
            print(f"  Mean |residual|:  {np.mean(valid_res):.2f}s  "
                  f"(median {np.median(valid_res):.2f}s)")
        print(f"  Result: {'PASS ✓' if good else 'FAIL ✗'}  "
              f"(alignment {'improved' if good else 'did not improve'} BPC correlation)")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT alignment quality test")
    parser.add_argument("--samples",   type=int,   default=8)
    parser.add_argument("--compounds", type=int,   default=60)
    parser.add_argument("--max-shift", type=float, default=30.0)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    run_test(
        n_samples=args.samples,
        n_compounds=args.compounds,
        max_shift_s=args.max_shift,
        seed=args.seed,
    )
