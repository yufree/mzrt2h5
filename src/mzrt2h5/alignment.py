"""
RT alignment module — streaming, QC-aware.

Workflow:
1. Single streaming pass over the H5 file to build a BPC (base peak
   chromatogram) for every sample. Memory: n_samples × n_bins × 4 bytes ≈ 10 MB.
2. Use the QC sample whose TIC is closest to the median QC TIC as the
   reference; run segmented cross-correlation to estimate per-sample local RT
   offsets (spline / linear interpolation).
3. Pre-compute an (n_samples × max_rt_bin) offset look-up table.
4. Second streaming pass to rewrite rt_indices in-place — no full load needed.

Design goals:
- QC samples as alignment reference (standard metabolomics practice).
- Large-file friendly: a 5–10 GB H5 runs on an 8 GB machine.
- Writes rt_aligned=True to H5 attributes so downstream tools can skip
  re-alignment.
"""
import numpy as np
import h5py
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm


DEFAULT_CHUNK_SIZE = 50_000_000   # ~200 MB per chunk, safe for 8 GB machines


def _build_all_bpc_streaming(h5_path, bin_size_s=1.0,
                              chunk_size=DEFAULT_CHUNK_SIZE):
    """Single streaming pass that builds a BPC for every sample simultaneously.

    Args:
        h5_path:     Path to the HDF5 file.
        bin_size_s:  BPC bin width in seconds (default 1 s).
        chunk_size:  Number of data points read per chunk.

    Returns:
        all_bpc:          ndarray (n_samples, n_bins) float32 — BPC matrix.
        storage_rt_prec:  float — H5 RT precision (seconds per index unit).
        n_bins:           int — number of BPC bins.
        num_samples:      int
        sample_names:     list[str]
    """
    with h5py.File(h5_path, 'r') as f:
        storage_rt_prec = float(f.attrs['rt_precision'])
        rt_max          = int(f['shape'][0])
        sample_ids_raw  = f['sample_id'][:]
        total_points    = len(f['data'])

    num_samples  = len(sample_ids_raw)
    sample_names = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in sample_ids_raw]

    bin_width = bin_size_s / storage_rt_prec   # bin 宽度（h5 索引单位）
    n_bins    = int(np.ceil(rt_max / bin_width)) + 1

    all_bpc = np.zeros((num_samples, n_bins), dtype=np.float32)

    with h5py.File(h5_path, 'r') as f:
        n_chunks = (total_points + chunk_size - 1) // chunk_size
        for start in tqdm(range(0, total_points, chunk_size),
                          desc="BPC scan", total=n_chunks):
            end      = min(start + chunk_size, total_points)
            c_data   = f['data'][start:end].astype(np.float32)
            c_rt     = f['rt_indices'][start:end]
            c_sample = f['sample_indices'][start:end].astype(np.int32)

            c_bin = np.clip((c_rt / bin_width).astype(np.int32), 0, n_bins - 1)

            # vectorised per-(sample, bin) max
            combined = c_sample.astype(np.int64) * n_bins + c_bin
            sort_idx = np.lexsort((-c_data, combined))
            s_comb   = combined[sort_idx]
            s_data   = c_data[sort_idx]

            uniq_keys, first = np.unique(s_comb, return_index=True)
            u_data   = s_data[first]
            u_sid    = (uniq_keys // n_bins).astype(np.int32)
            u_bin    = (uniq_keys %  n_bins).astype(np.int32)

            improve = u_data > all_bpc[u_sid, u_bin]
            if np.any(improve):
                all_bpc[u_sid[improve], u_bin[improve]] = u_data[improve]

    return all_bpc, storage_rt_prec, n_bins, num_samples, sample_names


def _segment_xcorr(ref_bpc, query_bpc, segment_bins=60, max_lag_bins=30):
    """Segmented normalised cross-correlation to estimate local RT offsets.

    Args:
        ref_bpc:       Reference BPC array.
        query_bpc:     Query BPC array.
        segment_bins:  Number of bins per segment.
        max_lag_bins:  Maximum search lag in bins.

    Returns:
        centers:  Segment centre positions (bin index, float).
        shifts:   Best-lag per segment (bins; positive = query shifted right
                  relative to reference).
    """
    n = min(len(ref_bpc), len(query_bpc))
    segments = []
    if n < segment_bins:
        segments = [(0, n)]
    else:
        for s in range(0, n - segment_bins // 2, segment_bins):
            e = min(s + segment_bins, n)
            if e - s >= segment_bins // 2:
                segments.append((s, e))

    centers = []
    shifts  = []

    for s, e in segments:
        ref_seg  = ref_bpc[s:e]
        q_s      = max(0, s - max_lag_bins)
        q_e      = min(n, e + max_lag_bins)
        q_wide   = query_bpc[q_s:q_e]

        centers.append((s + e) / 2.0)

        if ref_seg.sum() == 0 or q_wide.sum() == 0:
            shifts.append(0.0)
            continue

        ref_norm = ref_seg - ref_seg.mean()
        ref_std  = ref_seg.std()
        if ref_std == 0:
            shifts.append(0.0)
            continue

        seg_len  = e - s
        best_lag = 0
        best_corr = -1.0

        for lag in range(-max_lag_bins, max_lag_bins + 1):
            qs = (s + lag) - q_s
            qe = qs + seg_len
            if qs < 0 or qe > len(q_wide):
                continue
            q_seg  = q_wide[qs:qe]
            q_norm = q_seg - q_seg.mean()
            q_std  = q_seg.std()
            if q_std == 0:
                continue
            corr = np.dot(ref_norm, q_norm) / (ref_std * q_std * seg_len)
            if corr > best_corr:
                best_corr = corr
                best_lag  = lag

        # best_lag > 0 means query is shifted right by best_lag bins relative
        # to ref, so we correct by -best_lag.
        shifts.append(float(-best_lag))

    return np.array(centers, dtype=np.float32), np.array(shifts, dtype=np.float32)


def _shifts_to_correction(centers_bin, shifts_bin, rt_max_bin, max_shift_bin=None):
    """Fit discrete shift points into a continuous correction curve.

    Args:
        centers_bin:   Segment centres (H5 index units).
        shifts_bin:    Shift values (H5 index units).
        rt_max_bin:    Maximum RT index in the H5 file.
        max_shift_bin: Maximum allowed shift (H5 index units). Values outside
                       this range are clipped to suppress spline overshoot.

    Returns:
        rt_grid:    float32 array of H5 index coordinates.
        shift_vals: float32 array of corresponding shift values (H5 index units).
    """
    if len(centers_bin) == 0:
        return np.array([0.0, float(rt_max_bin)], dtype=np.float32), \
               np.array([0.0, 0.0], dtype=np.float32)

    if len(centers_bin) >= 4:
        try:
            spl = UnivariateSpline(
                centers_bin, shifts_bin,
                k=min(3, len(centers_bin) - 1),
                s=len(centers_bin) * 0.25)
            rt_grid    = np.linspace(0, float(rt_max_bin), 200)
            shift_vals = spl(rt_grid).astype(np.float32)
            # Clamp extrapolation at both ends to suppress boundary oscillation.
            shift_vals[rt_grid < centers_bin[0]]  = shifts_bin[0]
            shift_vals[rt_grid > centers_bin[-1]] = shifts_bin[-1]
            if max_shift_bin is not None:
                shift_vals = np.clip(shift_vals, -max_shift_bin, max_shift_bin)
            return rt_grid.astype(np.float32), shift_vals
        except Exception:
            pass

    if len(centers_bin) >= 2:
        rt_grid    = np.linspace(0, float(rt_max_bin), 200)
        shift_vals = np.interp(rt_grid, centers_bin, shifts_bin).astype(np.float32)
        if max_shift_bin is not None:
            shift_vals = np.clip(shift_vals, -max_shift_bin, max_shift_bin)
        return rt_grid.astype(np.float32), shift_vals

    med = float(np.median(shifts_bin))
    if max_shift_bin is not None:
        med = float(np.clip(med, -max_shift_bin, max_shift_bin))
    return (np.array([0.0, float(rt_max_bin)], dtype=np.float32),
            np.array([med, med], dtype=np.float32))


def compute_rt_corrections(h5_path,
                           qc_sample_names=None,
                           bin_size_s=1.0,
                           segment_size_s=60.0,
                           max_shift_s=30.0,
                           chunk_size=DEFAULT_CHUNK_SIZE):
    """Compute per-sample RT correction curves using BPC cross-correlation.

    Uses QC samples (or the highest-TIC sample when no QC list is provided) as
    the alignment reference.

    Args:
        h5_path:          Path to the HDF5 file.
        qc_sample_names:  List of QC sample names matching H5 sample_id values.
                          None → highest-TIC sample is used as reference.
        bin_size_s:       BPC bin width in seconds (default 1 s).
        segment_size_s:   Cross-correlation segment length in seconds (default 60 s).
        max_shift_s:      Maximum search lag in seconds (default 30 s).
        chunk_size:       Streaming read chunk size.

    Returns:
        dict with keys:
          'ref_sample_idx':  int — H5 index of the reference sample.
          'ref_sample_name': str — name of the reference sample.
          'corrections':     list of (rt_grid, shift_vals) tuples per sample;
                             both arrays are float32 in H5 RT index units.
          'median_shifts_s': list[float] — median shift per sample in seconds.
    """
    print("RT alignment: building BPC for all samples (streaming)...")
    all_bpc, storage_rt_prec, n_bins, num_samples, sample_names = \
        _build_all_bpc_streaming(h5_path, bin_size_s, chunk_size)

    bpc_tic   = all_bpc.sum(axis=1)
    bin_width = bin_size_s / storage_rt_prec
    seg_bins  = max(4, int(segment_size_s / bin_size_s))
    lag_bins  = max(2, int(max_shift_s   / bin_size_s))
    rt_max    = n_bins * bin_width

    name_to_idx = {n: i for i, n in enumerate(sample_names)}

    if qc_sample_names:
        qc_indices = [name_to_idx[n] for n in qc_sample_names
                      if n in name_to_idx]
        if len(qc_indices) == 0:
            print("  Warning: no QC samples matched in H5; "
                  "falling back to global TIC maximum")
            ref_idx = int(np.argmax(bpc_tic))
        else:
            qc_tics     = bpc_tic[qc_indices]
            median_tic  = np.median(qc_tics)
            rel_idx     = int(np.argmin(np.abs(qc_tics - median_tic)))
            ref_idx     = qc_indices[rel_idx]
            print(f"  {len(qc_indices)} QC samples found; "
                  f"reference = '{sample_names[ref_idx]}' "
                  f"(median QC TIC = {median_tic:.2e})")
    else:
        ref_idx = int(np.argmax(bpc_tic))
        print(f"  No QC list provided; reference = "
              f"'{sample_names[ref_idx]}' (highest TIC = {bpc_tic[ref_idx]:.2e})")

    ref_bpc   = all_bpc[ref_idx]
    ref_tic   = float(bpc_tic[ref_idx])
    # Cross-correlation is unreliable for near-blank samples; skip when TIC < 5 % of reference.
    min_tic   = ref_tic * 0.05

    corrections   = []
    median_shifts = []
    n_skipped     = 0

    zero_correction = (np.array([0.0, rt_max], dtype=np.float32),
                       np.array([0.0, 0.0],   dtype=np.float32))

    for sid in range(num_samples):
        if sid == ref_idx:
            corrections.append(zero_correction)
            median_shifts.append(0.0)
            continue

        if bpc_tic[sid] < min_tic:
            corrections.append(zero_correction)
            median_shifts.append(0.0)
            n_skipped += 1
            continue

        centers, seg_shifts = _segment_xcorr(
            ref_bpc, all_bpc[sid], seg_bins, lag_bins)

        centers_idx    = (centers    * bin_width).astype(np.float32)
        seg_shifts_idx = (seg_shifts * bin_width).astype(np.float32)

        max_shift_idx = float(lag_bins * bin_width)
        rt_grid, shift_vals = _shifts_to_correction(
            centers_idx, seg_shifts_idx, int(rt_max),
            max_shift_bin=max_shift_idx)

        corrections.append((rt_grid, shift_vals))
        med_s = float(np.median(seg_shifts)) * bin_size_s
        median_shifts.append(med_s)

    if n_skipped:
        print(f"  {n_skipped} low-TIC samples skipped (TIC < {min_tic:.1e}); "
              f"zero correction applied")

    nonzero = [abs(s) for s in median_shifts if s != 0]
    if nonzero:
        print(f"  Shift summary: median |shift| = {np.median(nonzero):.2f}s, "
              f"max = {max(nonzero):.2f}s  "
              f"({sum(1 for s in nonzero if abs(s) > 1.0)} samples shifted >1s)")

    return {
        'ref_sample_idx':  ref_idx,
        'ref_sample_name': sample_names[ref_idx],
        'corrections':     corrections,
        'median_shifts_s': median_shifts,
        'max_shift_bins':  float(lag_bins * bin_width),
    }


def apply_rt_corrections(h5_path, corrections_dict,
                         chunk_size=DEFAULT_CHUNK_SIZE,
                         inplace=True, output_path=None):
    """Apply RT corrections to an HDF5 file.

    Reads rt_indices in streaming chunks, adds per-sample offsets, and writes
    the corrected values back to the file. Supports in-place modification or
    writing to a new file.

    Args:
        h5_path:          Path to the input HDF5 file.
        corrections_dict: Return value of ``compute_rt_corrections()``.
        chunk_size:       Streaming chunk size (number of data points).
        inplace:          True → rewrite rt_indices in-place (no data copy).
        output_path:      Destination path when inplace=False; None overwrites
                          the source file.

    Returns:
        str: Path to the file that was written.
    """
    corrections = corrections_dict['corrections']
    num_samples = len(corrections)

    with h5py.File(h5_path, 'r') as f:
        rt_max = int(f['shape'][0])

    # Pre-compute full-resolution offset look-up table: (num_samples × rt_max) float32.
    # Typical footprint: 236 × 9597 × 4 bytes ≈ 9 MB.
    rt_axis        = np.arange(rt_max, dtype=np.float32)
    offset_table   = np.zeros((num_samples, rt_max), dtype=np.float32)
    for sid, (rt_grid, shift_vals) in enumerate(corrections):
        if np.all(shift_vals == 0):
            continue
        offset_table[sid] = np.interp(rt_axis, rt_grid, shift_vals,
                                      left=float(shift_vals[0]),
                                      right=float(shift_vals[-1]))

    # Final safety clip against spline oscillation producing extreme offsets.
    max_shift_bins = corrections_dict.get('max_shift_bins', None)
    if max_shift_bins is not None and max_shift_bins > 0:
        offset_table = np.clip(offset_table, -max_shift_bins, max_shift_bins)

    max_abs = np.abs(offset_table).max()
    with h5py.File(h5_path, 'r') as _f:
        _storage_rt_prec = float(_f.attrs.get('rt_precision', 0.1))
    print(f"  Offset table built: max correction = {max_abs:.1f} bins "
          f"= {max_abs * _storage_rt_prec:.2f}s  "
          f"(at {_storage_rt_prec}s storage precision)")

    if inplace:
        # Streaming in-place rewrite of rt_indices.
        with h5py.File(h5_path, 'r+') as f:
            total_points = len(f['data'])
            n_chunks = (total_points + chunk_size - 1) // chunk_size
            for start in tqdm(range(0, total_points, chunk_size),
                              desc="Applying RT correction", total=n_chunks):
                end       = min(start + chunk_size, total_points)
                c_rt      = f['rt_indices'][start:end]
                c_sample  = f['sample_indices'][start:end].astype(np.int32)

                safe_sid  = np.clip(c_sample, 0, num_samples - 1)
                safe_rt   = np.clip(c_rt, 0, rt_max - 1)
                offsets   = offset_table[safe_sid, safe_rt]

                corrected = np.clip(
                    np.round(c_rt.astype(np.float32) + offsets).astype(np.int32),
                    0, rt_max - 1)
                f['rt_indices'][start:end] = corrected

            f.attrs['rt_aligned']          = True
            f.attrs['rt_aligned_ref_idx']  = corrections_dict['ref_sample_idx']
            f.attrs['rt_aligned_ref_name'] = corrections_dict['ref_sample_name']

        return h5_path

    else:
        # Write a new file: copy all datasets, substitute corrected rt_indices.
        if output_path is None:
            output_path = h5_path

        with h5py.File(h5_path, 'r') as src, \
             h5py.File(output_path, 'w') as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            total_points = len(src['data'])
            n_chunks = (total_points + chunk_size - 1) // chunk_size

            dst.create_dataset('data',           data=src['data'][:],
                               compression='lzf', chunks=True)
            dst.create_dataset('mz_indices',     data=src['mz_indices'][:],
                               compression='lzf', chunks=True)
            dst.create_dataset('sample_indices', data=src['sample_indices'][:],
                               compression='lzf', chunks=True)

            rt_ds = dst.create_dataset(
                'rt_indices', shape=(total_points,), dtype=np.int32,
                compression='lzf', chunks=True)
            for start in tqdm(range(0, total_points, chunk_size),
                              desc="Writing corrected rt_indices", total=n_chunks):
                end      = min(start + chunk_size, total_points)
                c_rt     = src['rt_indices'][start:end]
                c_sample = src['sample_indices'][start:end].astype(np.int32)
                safe_sid = np.clip(c_sample, 0, num_samples - 1)
                safe_rt  = np.clip(c_rt,     0, rt_max - 1)
                offsets  = offset_table[safe_sid, safe_rt]
                corrected = np.clip(
                    np.round(c_rt.astype(np.float32) + offsets).astype(np.int32),
                    0, rt_max - 1)
                rt_ds[start:end] = corrected

            skip = {'data', 'rt_indices', 'mz_indices', 'sample_indices'}
            for key in src.keys():
                if key not in skip:
                    dst.create_dataset(key, data=src[key][:])

            dst.attrs['rt_aligned']          = True
            dst.attrs['rt_aligned_ref_idx']  = corrections_dict['ref_sample_idx']
            dst.attrs['rt_aligned_ref_name'] = corrections_dict['ref_sample_name']

        return output_path


def align_rt(h5_path,
             qc_sample_names=None,
             output_path=None,
             bin_size_s=1.0,
             segment_size_s=60.0,
             max_shift_s=30.0,
             chunk_size=DEFAULT_CHUNK_SIZE):
    """One-stop RT alignment: compute corrections and apply them to the H5 file.

    Args:
        h5_path:          Path to the input HDF5 file.
        qc_sample_names:  QC sample name list (None → highest-TIC sample used).
        output_path:      Output path (None → in-place modification).
        bin_size_s:       BPC bin width in seconds.
        segment_size_s:   Cross-correlation segment length in seconds.
        max_shift_s:      Maximum allowed shift in seconds.
        chunk_size:       Streaming read chunk size.

    Returns:
        str: Path to the file that was written.
    """
    print(f"=== RT Alignment: {h5_path} ===")
    inplace = (output_path is None)

    corr = compute_rt_corrections(
        h5_path, qc_sample_names,
        bin_size_s, segment_size_s, max_shift_s, chunk_size)

    out = apply_rt_corrections(
        h5_path, corr,
        chunk_size=chunk_size,
        inplace=inplace,
        output_path=output_path)

    print(f"RT alignment complete → {out}")
    return out
