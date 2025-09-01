import torch
from torch.utils.data import Dataset
import h5py
import scipy.sparse as sparse
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
                 aug_mz_shift_ppm=0.0): # Max m/z shift in PPM
        
        self.h5_path = h5_path
        self.transform = transform
        self.augment = augment
        self.aug_rt_shift_s = aug_rt_shift_s
        self.aug_mz_shift_ppm = aug_mz_shift_ppm
        
        # --- 1. Load all data and metadata from HDF5 file ---
        with h5py.File(self.h5_path, 'r') as f:
            # Load the 3D-coordinate sparse data
            self.all_data = f['data'][:]
            self.all_rt_indices = f['rt_indices'][:]
            self.all_mz_indices = f['mz_indices'][:]
            self.all_sample_indices = f['sample_indices'][:]

            if min_intensity is not None and min_intensity > 0:
                intensity_mask = self.all_data > min_intensity
                self.all_data = self.all_data[intensity_mask]
                self.all_rt_indices = self.all_rt_indices[intensity_mask]
                self.all_mz_indices = self.all_mz_indices[intensity_mask]
                self.all_sample_indices = self.all_sample_indices[intensity_mask]
            # Load all covariate datasets
            self.covariates = {key: f[key][:] for key in f.keys() 
                               if key not in ['data', 'rt_indices', 'mz_indices', 'sample_indices', 'shape']}
            
            # Load all mappings from attributes
            self.mappings = {}
            if 'mappings' in f.attrs:
                self.mappings = json.loads(f.attrs['mappings'])

            # Load metadata for rescaling and cropping
            self.storage_shape = tuple(f['shape'][:])
            self.storage_rt_precision = f.attrs['rt_precision']
            self.storage_mz_precision = f.attrs['mz_precision']
            self.storage_rt_range = (f.attrs['rt_range_min'], f.attrs['rt_range_max'])
            self.storage_mz_range = (f.attrs['mz_range_min'], f.attrs['mz_range_max'])

        # --- 2. Pre-calculate slices for fast sample lookup ---
        self.num_total_samples = 0
        if len(self.all_sample_indices) > 0:
            self.num_total_samples = np.max(self.all_sample_indices) + 1
            
        self.sample_slices = {}
        boundaries = np.searchsorted(self.all_sample_indices, np.arange(self.num_total_samples + 1))
        for i in range(self.num_total_samples):
            self.sample_slices[i] = (boundaries[i], boundaries[i+1])

        # --- 3. Validate target_covariate ---
        if target_covariate not in self.covariates:
            raise ValueError(f"Error: Specified target_covariate '{target_covariate}' not in HDF5 file. "
                             f"Available covariates: {list(self.covariates.keys())}")
        self.target_covariate = target_covariate
        print(f"Dataset initialized. Prediction target is set to: '{self.target_covariate}'")

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

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        
        # --- A. Reconstruct High-Res Image for one sample ---
        start, end = self.sample_slices[actual_idx]
        sample_data = self.all_data[start:end]
        sample_rt_indices = self.all_rt_indices[start:end]
        sample_mz_indices = self.all_mz_indices[start:end]
        
        hr_sparse_matrix = sparse.coo_matrix(
            (sample_data, (sample_rt_indices, sample_mz_indices)),
            shape=self.storage_shape
        )
        
        # --- B. Rescale to Target Resolution ---
        lr_row_indices = np.floor(hr_sparse_matrix.row / self.rt_scaling_factor).astype(int)
        lr_col_indices = np.floor(hr_sparse_matrix.col / self.mz_scaling_factor).astype(int)

        # --- C. Apply Augmentation (if enabled) ---
        if self.augment:
            # RT shift
            rt_shift_s = np.random.uniform(-self.aug_rt_shift_s, self.aug_rt_shift_s)
            rt_shift_pixels = int(rt_shift_s / self.target_rt_precision)
            lr_row_indices += rt_shift_pixels

            # m/z shift (approximated)
            mz_shift_ppm = np.random.uniform(-self.aug_mz_shift_ppm, self.aug_mz_shift_ppm)
            center_mz = np.mean(self.storage_mz_range)
            mz_delta = center_mz * mz_shift_ppm * 1e-6
            mz_shift_pixels = int(mz_delta / self.target_mz_precision)
            lr_col_indices += mz_shift_pixels

        # --- D. Clip indices to prevent out-of-bounds error ---
        h, w = self.target_shape
        lr_row_indices = np.clip(lr_row_indices, 0, h - 1)
        lr_col_indices = np.clip(lr_col_indices, 0, w - 1)

        lr_sparse = sparse.coo_matrix(
            (hr_sparse_matrix.data, (lr_row_indices, lr_col_indices)),
            shape=self.target_shape
        )
        image_tensor = torch.from_numpy(lr_sparse.toarray()).unsqueeze(0).float()
        
        # --- E. Apply Spatial Crop ---
        if self.crop_slice:
            image_tensor = image_tensor[:, self.crop_slice[0], self.crop_slice[1]]
        
        # --- F. Get the Target Label ---
        labels_dict = {}
        for key, values in self.covariates.items():
            value = values[actual_idx]
            map_name = f"{key}_to_idx"
            if map_name in self.mappings:
                value_str = value.decode('utf-8')
                labels_dict[key] = torch.tensor(self.mappings[map_name][value_str])
            else:
                labels_dict[key] = torch.tensor(value, dtype=torch.float32)

        final_label = labels_dict[self.target_covariate]
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, final_label.long()
