import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import json

class DynamicSparseH5Dataset(Dataset):
    """
    The final, robust PyTorch Dataset for loading from a self-contained, sparse HDF5 file.
    
    This class correctly implements:
    1. Loading from the 3D-coordinate sparse format.
    2. Dynamic rescaling to a target resolution.
    3. Filtering of samples based on covariate values.
    4. Cropping of the final image to a specific RT/m/z range.
    5. Selection of a specific covariate as the prediction target.
    6. On-the-fly data augmentation with physically meaningful units.
    """
    def __init__(self, h5_path, target_rt_precision, target_mz_precision,
                 target_covariate='class',
                 min_intensity=None,
                 covariate_filters=None,
                 crop_rt_range=None,
                 crop_mz_range=None,
                 transform=None,
                 augment=False,
                 aug_rt_shift_s=0.0,  # Max RT shift in seconds
                 aug_mz_shift_ppm=0.0, # Max m/z shift in PPM
                 apply_log1p_norm=False, # Apply log1p + MinMax norm
                 cache=True): # Precompute & cache each sample's LR image once
        
        self.h5_path = h5_path
        self.transform = transform
        self.augment = augment
        self.aug_rt_shift_s = aug_rt_shift_s
        self.aug_mz_shift_ppm = aug_mz_shift_ppm
        self.apply_log1p_norm = apply_log1p_norm
        
        # --- 1. Read metadata + per-sample slice boundaries (NOT the HR points) ---
        # The 3D sparse HR arrays (data/rt/mz/sample indices) are ~15 GB on real
        # data; loading them all just to downsample each sample to a small LR
        # image is what OOMs 16 GB machines. Instead we read only sample_indices
        # (in chunks → tiny peak) to learn where each sample's points live, then
        # read each sample's HR slice on demand in _compute_lr_image. Every use
        # of this class is "give me sample i's LR image", so the full HR set is
        # never needed in memory.
        self.min_intensity = min_intensity
        with h5py.File(self.h5_path, 'r') as f:
            self.covariates = {key: f[key][:] for key in f.keys()
                               if key not in ['data', 'rt_indices', 'mz_indices', 'sample_indices', 'shape']}
            self.mappings = {}
            if 'mappings' in f.attrs:
                self.mappings = json.loads(f.attrs['mappings'])
            self.storage_shape = tuple(f['shape'][:])
            self.storage_rt_precision = f.attrs['rt_precision']
            self.storage_mz_precision = f.attrs['mz_precision']
            self.storage_rt_range = (f.attrs['rt_range_min'], f.attrs['rt_range_max'])
            self.storage_mz_range = (f.attrs['mz_range_min'], f.attrs['mz_range_max'])

            n_total = int(f['sample_indices'].shape[0])
            # number of samples
            self.num_total_samples = 0
            if self.covariates:
                self.num_total_samples = len(next(iter(self.covariates.values())))
            elif n_total > 0:
                self.num_total_samples = int(f['sample_indices'][-1]) + 1

            # --- 2. Per-sample [start, end) boundaries via chunked scan ---
            # sample_indices is sorted ascending (mzrt2h5 writes per-sample), so
            # each sample id's points are a contiguous block. Find each block's
            # start without holding the whole array (peak = one chunk).
            starts = {}
            scan = 50_000_000
            for c0 in range(0, n_total, scan):
                si = f['sample_indices'][c0:c0 + scan]
                uniq, idx = np.unique(si, return_index=True)
                for u, i in zip(uniq.tolist(), idx.tolist()):
                    if u not in starts:
                        starts[u] = c0 + i
            self.sample_slices = {}
            for i in range(self.num_total_samples):
                s = starts.get(i, n_total)
                e = starts.get(i + 1, n_total)
                self.sample_slices[i] = (s, e)

        # persistent read handle (reused by _compute_lr_image; trainer uses
        # num_workers=0 so a single handle is safe). Opened lazily.
        self._read_f = None

        # --- 3. Validate target_covariate ---
        if target_covariate is not None and target_covariate not in self.covariates:
            raise ValueError(f"Error: Specified target_covariate '{target_covariate}' not in HDF5 file. "
                             f"Available covariates: {list(self.covariates.keys())}")
        self.target_covariate = target_covariate
        if target_covariate is not None:
            print(f"Dataset initialized. Prediction target is set to: '{self.target_covariate}'")
        else:
            print(f"Dataset initialized. Inference mode (no target covariate).")

        # --- 4. Setup Rescaling ---
        self.target_rt_precision = target_rt_precision
        self.target_mz_precision = target_mz_precision
        self.rt_scaling_factor = self.target_rt_precision / self.storage_rt_precision
        self.mz_scaling_factor = self.target_mz_precision / self.storage_mz_precision
        new_rt_size = int(self.storage_shape[0] / self.rt_scaling_factor)
        new_mz_size = int(self.storage_shape[1] / self.mz_scaling_factor)
        self.target_shape = (new_rt_size, new_mz_size)
        
        # --- 5. Apply Covariate Filters ---
        self.filtered_indices = list(range(self.num_total_samples))
        if covariate_filters:
            print("Applying covariate filters...")
            passing_indices = []
            for idx in self.filtered_indices:
                passes_all_filters = True
                for key, condition in covariate_filters.items():
                    value = self.covariates[key][idx]
                    if isinstance(value, bytes): value = value.decode('utf-8')
                    if callable(condition):
                        if not condition(value): passes_all_filters = False; break
                    elif str(value) != str(condition): # Use string comparison for robustness
                        passes_all_filters = False; break
                if passes_all_filters: passing_indices.append(idx)
            self.filtered_indices = passing_indices
            print(f"Filtering complete. {len(self.filtered_indices)} samples remaining.")

        # --- 6. Pre-calculate Crop Slice Indices ---
        self.crop_slice = None
        if crop_rt_range or crop_mz_range:
            h, w = self.target_shape
            rt_min_idx, rt_max_idx = 0, h
            mz_min_idx, mz_max_idx = 0, w
            if crop_rt_range: rt_min_idx = int((crop_rt_range[0] - self.storage_rt_range[0]) / self.target_rt_precision); rt_max_idx = int((crop_rt_range[1] - self.storage_rt_range[0]) / self.target_rt_precision)
            if crop_mz_range: mz_min_idx = int((crop_mz_range[0] - self.storage_mz_range[0]) / self.target_mz_precision); mz_max_idx = int((crop_mz_range[1] - self.storage_mz_range[0]) / self.target_mz_precision)
            rt_min_idx = max(0, rt_min_idx); rt_max_idx = min(h, rt_max_idx); mz_min_idx = max(0, mz_min_idx); mz_max_idx = min(w, mz_max_idx)
            self.crop_slice = (slice(rt_min_idx, rt_max_idx), slice(mz_min_idx, mz_max_idx))
            print(f"Cropping enabled. Slicing to RT pixels {rt_min_idx}:{rt_max_idx}, m/z pixels {mz_min_idx}:{mz_max_idx}.")

        # --- 7. Precompute & cache each sample's LR image (once) ---
        # The HR→LR rescaling for a sample is deterministic — it depends only on
        # the target resolution, not on the epoch. Rebuilding it from the sparse
        # HR points on every __getitem__ (the column dim is ~1e7 wide) costs
        # ~1.7 s/sample, which is fatal across training epochs. Compute the dense
        # LR image once here; __getitem__ then only does (cheap) augmentation /
        # crop / normalization on the cached array. Augmentation stays dynamic
        # because it is applied per access, not baked into the cache.
        self.cache = cache
        self._lr_cache = None
        if cache:
            self._lr_cache = {
                ai: self._compute_lr_image(ai) for ai in self.filtered_indices
            }

    def _compute_lr_image(self, actual_idx):
        """Read one sample's HR points from the H5 and rescale to the LR grid.

        Returns a dense (H, W) float32 ndarray on the *uncropped, un-augmented,
        un-normalized* target grid. Reads only this sample's slice (not the whole
        H5), so memory stays bounded regardless of file size."""
        start, end = self.sample_slices[actual_idx]
        if end <= start:
            return np.zeros(self.target_shape, dtype=np.float32)
        if self._read_f is None:
            self._read_f = h5py.File(self.h5_path, 'r')
        f = self._read_f
        sample_data = f['data'][start:end]
        sample_rt_indices = f['rt_indices'][start:end]
        sample_mz_indices = f['mz_indices'][start:end]

        if self.min_intensity is not None and self.min_intensity > 0:
            keep = sample_data > self.min_intensity
            sample_data = sample_data[keep]
            sample_rt_indices = sample_rt_indices[keep]
            sample_mz_indices = sample_mz_indices[keep]

        # Map HR coords -> LR grid and accumulate directly (avoid building a
        # full-width ~1e7-column HR coo_matrix just to read .row/.col).
        lr_row = np.floor(sample_rt_indices / self.rt_scaling_factor).astype(np.int64)
        lr_col = np.floor(sample_mz_indices / self.mz_scaling_factor).astype(np.int64)
        h, w = self.target_shape
        np.clip(lr_row, 0, h - 1, out=lr_row)
        np.clip(lr_col, 0, w - 1, out=lr_col)
        # Flatten to 1D and use bincount (vectorized) instead of np.add.at
        # (np.add.at is a Python-level scatter and is the slow part).
        flat = lr_row * w + lr_col
        img = np.bincount(flat, weights=sample_data, minlength=h * w)
        return img.reshape(h, w).astype(np.float32)

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]

        # --- A+B. LR image (cached if available, else compute once now) ---
        if self._lr_cache is not None:
            lr_img = self._lr_cache[actual_idx]
        else:
            lr_img = self._compute_lr_image(actual_idx)

        # --- C. Apply Augmentation (if enabled) ---
        # Augmentation is per-access (keeps the cache clean): integer pixel
        # shifts on the dense LR image == the old coordinate shift, but cheap.
        if self.augment:
            rt_shift_pixels = int(np.random.uniform(-self.aug_rt_shift_s, self.aug_rt_shift_s)
                                  / self.target_rt_precision)
            center_mz = np.mean(self.storage_mz_range)
            mz_delta = center_mz * np.random.uniform(-self.aug_mz_shift_ppm, self.aug_mz_shift_ppm) * 1e-6
            mz_shift_pixels = int(mz_delta / self.target_mz_precision)
            if rt_shift_pixels or mz_shift_pixels:
                lr_img = np.roll(lr_img, (rt_shift_pixels, mz_shift_pixels), axis=(0, 1))

        image_tensor = torch.from_numpy(np.ascontiguousarray(lr_img)).unsqueeze(0).float()

        # --- E. Apply Spatial Crop ---
        if self.crop_slice:
            image_tensor = image_tensor[:, self.crop_slice[0], self.crop_slice[1]]
            
        # --- E2. Apply Log Transformation if enabled ---
        if self.apply_log1p_norm:
            image_tensor = torch.log1p(image_tensor)
            img_min = image_tensor.min()
            img_max = image_tensor.max()
            if img_max - img_min > 0:
                image_tensor = (image_tensor - img_min) / (img_max - img_min + 1e-6)
        
        # --- F. Get the Target Label ---
        if self.target_covariate is None:
            final_label = torch.tensor(0, dtype=torch.long)
        else:
            labels_dict = {}
            for key, values in self.covariates.items():
                value = values[actual_idx]
                map_name = f"{key}_to_idx"
                if map_name in self.mappings:
                    value_str = value.decode('utf-8') if isinstance(value, bytes) else str(value)
                    labels_dict[key] = torch.tensor(self.mappings[map_name].get(value_str, 0))
                else:
                    try:
                        labels_dict[key] = torch.tensor(value, dtype=torch.float32)
                    except (TypeError, ValueError):
                        # Fallback for non-numeric data without mapping
                        if isinstance(value, bytes):
                            try:
                                labels_dict[key] = torch.tensor(float(value.decode('utf-8')), dtype=torch.float32)
                            except (TypeError, ValueError):
                                labels_dict[key] = torch.tensor(0.0, dtype=torch.float32)
                        else:
                            labels_dict[key] = torch.tensor(0.0, dtype=torch.float32)
    
            final_label = labels_dict[self.target_covariate].long()
        
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, final_label

    def __del__(self):
        # close the lazy read handle if open
        f = getattr(self, "_read_f", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
