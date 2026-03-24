"""
样本间 RT 对齐模块

基于 base peak chromatogram (BPC) 互相关的轻量级 RT 校正：
1. 每个样本按 1s RT bin 切条，取每条的最大强度离子 → BPC
2. 分段互相关计算局部 RT 偏移
3. 样条插值得到连续偏移曲线
4. 对 h5 中的 rt_indices 应用校正

设计目标：快速、覆盖全 RT 范围、适用于 Master Image 构建前的预处理
"""
import numpy as np
import h5py
from scipy.interpolate import UnivariateSpline


def _build_bpc(data, rt_indices, sample_indices, sample_id,
               storage_rt_precision, bin_size_s=1.0):
    """构建单个样本的 base peak chromatogram (BPC)

    将 RT 轴按 bin_size_s 切 bin，每个 bin 取最大强度值。

    Args:
        data: 强度数组
        rt_indices: h5 RT 索引数组
        sample_indices: 样本索引数组
        sample_id: 目标样本 ID
        storage_rt_precision: h5 存储的 RT 精度（秒/索引）
        bin_size_s: BPC bin 宽度（秒）

    Returns:
        bpc: 1D 数组，每个元素是该 RT bin 的最大强度
        bin_edges_idx: bin 边界（h5 索引单位）
    """
    mask = sample_indices == sample_id
    if not np.any(mask):
        return np.array([]), np.array([])

    s_data = data[mask].astype(np.float64)
    s_rt = rt_indices[mask]

    # h5 索引转 bin
    bin_width = bin_size_s / storage_rt_precision
    rt_max = int(rt_indices.max())
    n_bins = int(np.ceil(rt_max / bin_width)) + 1

    bpc = np.zeros(n_bins, dtype=np.float64)
    bin_idx = (s_rt / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # 每个 bin 取 max（用 ufunc.at 避免循环）
    np.maximum.at(bpc, bin_idx, s_data)

    return bpc


def _segment_xcorr(ref_bpc, query_bpc, segment_size=60, max_lag=30):
    """分段互相关计算局部 RT 偏移

    将 BPC 分成若干段，每段做互相关找最佳偏移。

    Args:
        ref_bpc: 参考样本 BPC
        query_bpc: 查询样本 BPC
        segment_size: 每段长度（bin 数，即秒数 if bin=1s）
        max_lag: 最大搜索偏移（bin 数）

    Returns:
        segment_centers: 每段的中心位置（bin index）
        segment_shifts: 每段的最佳偏移量（bin 数，正=query 右移到 ref）
    """
    n = min(len(ref_bpc), len(query_bpc))
    if n < segment_size:
        # 全局互相关
        segments = [(0, n)]
    else:
        segments = []
        for start in range(0, n - segment_size // 2, segment_size):
            end = min(start + segment_size, n)
            if end - start >= segment_size // 2:
                segments.append((start, end))

    centers = []
    shifts = []

    for start, end in segments:
        ref_seg = ref_bpc[start:end]
        # query 取更宽的范围用于搜索
        q_start = max(0, start - max_lag)
        q_end = min(n, end + max_lag)
        query_wide = query_bpc[q_start:q_end]

        if np.sum(ref_seg) == 0 or np.sum(query_wide) == 0:
            centers.append((start + end) / 2.0)
            shifts.append(0.0)
            continue

        # 归一化互相关
        ref_norm = ref_seg - ref_seg.mean()
        ref_std = np.std(ref_seg)
        if ref_std == 0:
            centers.append((start + end) / 2.0)
            shifts.append(0.0)
            continue

        best_corr = -1
        best_lag = 0
        seg_len = end - start

        for lag in range(-max_lag, max_lag + 1):
            # query 段在 wide 数组中的位置
            qs = (start + lag) - q_start
            qe = qs + seg_len
            if qs < 0 or qe > len(query_wide):
                continue
            q_seg = query_wide[qs:qe]
            q_norm = q_seg - q_seg.mean()
            q_std = np.std(q_seg)
            if q_std == 0:
                continue
            corr = np.dot(ref_norm, q_norm) / (ref_std * q_std * seg_len)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        centers.append((start + end) / 2.0)
        # best_lag > 0 表示 query 右移了，校正需要左移（负值）
        shifts.append(float(-best_lag))

    return np.array(centers), np.array(shifts)


def compute_rt_shifts(h5_path, bin_size_s=1.0, segment_size_s=60,
                      max_shift_s=30, ref_sample=None):
    """计算每个样本相对于参考的 RT 偏移曲线

    基于 BPC 分段互相关：
    1. 每个样本构建 1s bin 的 BPC（base peak chromatogram）
    2. 分 60s 段做互相关，得到局部偏移
    3. 样条插值得到连续偏移曲线

    Args:
        h5_path: HDF5 文件路径
        bin_size_s: BPC bin 宽度（秒），默认 1s
        segment_size_s: 互相关段长度（秒），默认 60s
        max_shift_s: 最大搜索偏移（秒），默认 30s
        ref_sample: 参考样本索引（None 则自动选 TIC 最高的）

    Returns:
        dict: {
            'ref_sample': 参考样本索引,
            'shifts': list of (rt_grid, shift_values) per sample,
                      rt_grid 和 shift_values 单位均为 h5 索引,
            'median_shifts_s': list 每个样本的中位偏移（秒）,
        }
    """
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
        rt_idx = f['rt_indices'][:]
        sample_idx = f['sample_indices'][:]
        storage_rt_prec = f.attrs['rt_precision']

    num_samples = int(sample_idx.max()) + 1
    bin_width_idx = bin_size_s / storage_rt_prec  # bin 宽度（h5 索引单位）
    segment_bins = int(segment_size_s / bin_size_s)
    max_lag_bins = int(max_shift_s / bin_size_s)
    rt_max = int(rt_idx.max())

    # 构建所有样本的 BPC
    all_bpc = []
    bpc_sums = []
    for sid in range(num_samples):
        bpc = _build_bpc(data, rt_idx, sample_idx, sid,
                         storage_rt_prec, bin_size_s)
        all_bpc.append(bpc)
        bpc_sums.append(np.sum(bpc))

    # 选参考样本：TIC 最高的
    if ref_sample is None:
        ref_sample = int(np.argmax(bpc_sums))
    ref_bpc = all_bpc[ref_sample]
    print(f"Reference sample: {ref_sample} (TIC = {bpc_sums[ref_sample]:.0f})")

    shifts = []
    median_shifts_s = []

    for sid in range(num_samples):
        if sid == ref_sample:
            shifts.append((np.array([0.0, float(rt_max)]), np.array([0.0, 0.0])))
            median_shifts_s.append(0.0)
            continue

        # 分段互相关
        centers, seg_shifts = _segment_xcorr(
            ref_bpc, all_bpc[sid], segment_bins, max_lag_bins)

        if len(centers) == 0:
            shifts.append((np.array([0.0, float(rt_max)]), np.array([0.0, 0.0])))
            median_shifts_s.append(0.0)
            continue

        # bin 单位 → h5 索引单位
        centers_idx = centers * bin_width_idx
        shifts_idx = seg_shifts * bin_width_idx

        # 样条插值得到连续偏移曲线
        if len(centers_idx) >= 4:
            try:
                spline = UnivariateSpline(
                    centers_idx, shifts_idx,
                    k=min(3, len(centers_idx) - 1),
                    s=len(centers_idx) * (bin_width_idx * 0.5) ** 2)
                rt_grid = np.linspace(0, float(rt_max), 200)
                shift_vals = spline(rt_grid)
            except Exception:
                median_s = float(np.median(shifts_idx))
                rt_grid = np.array([0.0, float(rt_max)])
                shift_vals = np.array([median_s, median_s])
        elif len(centers_idx) >= 2:
            # 线性插值
            rt_grid = np.linspace(0, float(rt_max), 200)
            shift_vals = np.interp(rt_grid, centers_idx, shifts_idx)
        else:
            median_s = float(np.median(shifts_idx))
            rt_grid = np.array([0.0, float(rt_max)])
            shift_vals = np.array([median_s, median_s])

        shifts.append((rt_grid, shift_vals))
        med_s = float(np.median(seg_shifts)) * bin_size_s
        median_shifts_s.append(med_s)
        print(f"  Sample {sid}: median shift = {med_s:+.1f}s, "
              f"range = {seg_shifts.min() * bin_size_s:.1f}~{seg_shifts.max() * bin_size_s:.1f}s")

    return {
        'ref_sample': ref_sample,
        'shifts': shifts,
        'median_shifts_s': median_shifts_s,
    }


def align_h5(h5_path, output_path=None, bin_size_s=1.0,
             segment_size_s=60, max_shift_s=30, ref_sample=None):
    """对 h5 文件做 RT 对齐，输出新的对齐后 h5 文件

    Args:
        h5_path: 输入 HDF5 文件路径
        output_path: 输出路径（None 则覆盖原文件）
        bin_size_s: BPC bin 宽度（秒）
        segment_size_s: 互相关段长度（秒）
        max_shift_s: 最大搜索偏移（秒）
        ref_sample: 参考样本索引

    Returns:
        dict: 对齐统计信息
    """
    alignment = compute_rt_shifts(h5_path, bin_size_s, segment_size_s,
                                  max_shift_s, ref_sample)

    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:]
        rt_idx = f['rt_indices'][:]
        mz_idx = f['mz_indices'][:]
        sample_idx = f['sample_indices'][:]
        shape = f['shape'][:]
        attrs = dict(f.attrs)
        other_datasets = {}
        for key in f.keys():
            if key not in ['data', 'rt_indices', 'mz_indices',
                           'sample_indices', 'shape']:
                other_datasets[key] = f[key][:]

    storage_rt_prec = attrs['rt_precision']
    corrected_rt = rt_idx.astype(np.float64).copy()
    num_samples = int(sample_idx.max()) + 1

    for sid in range(num_samples):
        mask = sample_idx == sid
        if not np.any(mask):
            continue
        rt_grid, shift_vals = alignment['shifts'][sid]
        sample_rt = rt_idx[mask].astype(np.float64)
        shift = np.interp(sample_rt, rt_grid, shift_vals)
        corrected_rt[mask] = sample_rt + shift

    corrected_rt = np.clip(corrected_rt, 0, shape[0] - 1)
    corrected_rt_int = np.round(corrected_rt).astype(np.int32)

    if output_path is None:
        output_path = h5_path

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=data, compression='gzip')
        f.create_dataset('rt_indices', data=corrected_rt_int, compression='gzip')
        f.create_dataset('mz_indices', data=mz_idx, compression='gzip')
        f.create_dataset('sample_indices', data=sample_idx, compression='gzip')
        f.create_dataset('shape', data=shape)
        for key, val in other_datasets.items():
            f.create_dataset(key, data=val)
        for key, val in attrs.items():
            f.attrs[key] = val
        f.attrs['aligned'] = True
        f.attrs['alignment_ref_sample'] = alignment['ref_sample']

    avg_shift = np.mean([abs(s) for s in alignment['median_shifts_s']])
    print(f"\nAlignment complete → {output_path}")
    print(f"  Average |median shift|: {avg_shift:.1f}s")

    return alignment
