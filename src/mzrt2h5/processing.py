import os
import numpy as np
import pandas as pd
import pymzml
import scipy.sparse as sparse
import h5py
import json
from tqdm import tqdm

def load_metadata_from_file(file_path, sample_id_col, separator=','):
    """
    Loads metadata from a CSV or TSV file into a dictionary.

    Args:
        file_path (str): Path to the metadata file.
        sample_id_col (str): The column name containing the unique sample IDs.
        separator (str): The separator used in the file (e.g., ',', '\t'). 
                         Defaults to comma. Using a specific separator is more robust
                         than relying on whitespace.
    
    Returns:
        dict: A dictionary mapping sample IDs to their metadata.
    """
    try:
        # Using a specific separator like ',' or '\t' is generally safer than '\s+'
        df = pd.read_csv(file_path, sep=separator)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Metadata file not found at {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file with pandas: {e}")

    if sample_id_col not in df.columns:
        raise ValueError(f"Error: The specified sample ID column '{sample_id_col}' was not found in the metadata file.")
        
    # Convert all metadata to a dictionary for quick lookups
    metadata_lookup = df.set_index(sample_id_col).to_dict('index')    
    
    return metadata_lookup
    
def process_mzml_to_sparse(file, rt_precision, mz_precision, mz_range=None, rt_range=None):
    """
    Processes a single mzML file into a sparse 2D matrix (RT vs. m/z).

    Args:
        file (str): Path to the .mzML file.
        rt_precision (float): The bin size for the retention time axis.
        mz_precision (float): The bin size for the m/z axis.
        mz_range (tuple, optional): A (min, max) tuple to fix the m/z range.
        rt_range (tuple, optional): A (min, max) tuple to fix the RT range.

    Returns:
        tuple: A COO sparse matrix, the used RT range, and the used m/z range.
    """
    run = pymzml.run.Reader(file)
    
    spectra_data = []
    for spectrum in run:
        # Process only MS1 level scans with actual data points
        if spectrum.ms_level == 1 and len(spectrum.mz) > 0:
            spectra_data.append({
                "rt": spectrum.scan_time_in_minutes() * 60, # Convert RT to seconds
                "mz": spectrum.mz,
                "intensity": spectrum.i.astype(np.int32)
            })

    # If the file is empty or has no MS1 scans, return an empty matrix
    if not spectra_data:
        if rt_range and mz_range:
            shape = (int((rt_range[1] - rt_range[0]) / rt_precision) + 1, 
                     int((mz_range[1] - mz_range[0]) / mz_precision) + 1)
            return sparse.coo_matrix(shape), rt_range, mz_range
        else:
            # Cannot determine shape if no data and no ranges are provided
            return sparse.coo_matrix((0, 0)), (0, 0), (0, 0)

    # Determine ranges from data if not provided
    if rt_range is None:
        min_rt = min(s['rt'] for s in spectra_data)
        max_rt = max(s['rt'] for s in spectra_data)
        rt_range = (min_rt, max_rt)

    if mz_range is None:
        min_mz = min(np.min(s['mz']) for s in spectra_data)
        max_mz = max(np.max(s['mz']) for s in spectra_data)
        mz_range = (min_mz, max_mz)

    rt_min, rt_max = rt_range
    mz_min, mz_max = mz_range

    rt_size = int((rt_max - rt_min) / rt_precision) + 1
    mz_size = int((mz_max - mz_min) / mz_precision) + 1
    
    row_indices, col_indices, intensities = [], [], []
    for spec in spectra_data:
        rt, mz, intensity = spec['rt'], np.array(spec['mz']), np.array(spec['intensity'])
        
        # Filter scans outside the desired RT range
        if not (rt >= rt_min and rt <= rt_max):
            continue

        # Filter m/z values outside the desired m/z range
        idx_mz = (mz >= mz_min) & (mz <= mz_max)
        filtered_mz = mz[idx_mz]
        filtered_intensity = intensity[idx_mz]
        
        if len(filtered_mz) == 0:
            continue

        # Bin RT and m/z values to the specified precision
        # Note: This simple rounding can be replaced with more advanced binning if needed
        binned_rt = np.round(rt / rt_precision) * rt_precision
        binned_mz = np.round(filtered_mz / mz_precision) * mz_precision
        
        # Convert binned values to integer indices for the sparse matrix
        rt_idx = int((binned_rt - rt_min) / rt_precision)
        mz_indices = ((binned_mz - mz_min) / mz_precision).astype(int)

        rt_idx = np.clip(rt_idx, 0, rt_size - 1)
        mz_indices = np.clip(mz_indices, 0, mz_size - 1)

        row_indices.extend([rt_idx] * len(mz_indices))
        col_indices.extend(mz_indices)
        intensities.extend(filtered_intensity)

    # Define the final shape of the sparse matrix
    rt_size = int((rt_max - rt_min) / rt_precision) + 1
    mz_size = int((mz_max - mz_min) / mz_precision) + 1
    
    final_sparse_matrix = sparse.coo_matrix((intensities, (row_indices, col_indices)), shape=(rt_size, mz_size))
    
    return final_sparse_matrix, rt_range, mz_range

def save_dataset_as_sparse_h5(folder, save_path, rt_precision, mz_precision,
                              metadata_csv_path,      
                              sample_id_col='Sample Name',
                              separator=',',
                              mz_range=None, rt_range=None):
    """
    Processes a folder of mzML files and saves them as a single, consolidated
    sparse HDF5 file with associated metadata.

    Args:
        folder (str): Path to the folder containing .mzML files.
        save_path (str): Path to save the output .h5 file.
        rt_precision (float): Bin size for the retention time axis.
        mz_precision (float): Bin size for the m/z axis.
        metadata_csv_path (str): Path to the metadata CSV file.
        sample_id_col (str): Column name for sample IDs in the metadata.
        separator (str): Separator for the metadata file (e.g., ',').
        mz_range (tuple, optional): Fixed (min, max) m/z range.
        rt_range (tuple, optional): Fixed (min, max) RT range.
    """
    
    # Load metadata from the provided CSV file
    metadata_lookup = load_metadata_from_file(metadata_csv_path, sample_id_col, separator)
    print(f"Successfully loaded {len(metadata_lookup)} metadata records from {metadata_csv_path}.")

    files_to_process = []
    all_covariates = []
    
    # Find all .mzML files and match them with loaded metadata
    all_mzml_files = [os.path.join(root, f) for root, _, fs in os.walk(folder) for f in fs if f.endswith('.mzML')]
    
    for f_path in all_mzml_files:
        # Assumes filename (without extension) is the sample ID
        sample_id = os.path.basename(f_path).replace('.mzML', '')
        if sample_id in metadata_lookup:
            files_to_process.append(f_path)
            all_covariates.append(metadata_lookup[sample_id])
        else:
            print(f"Warning: Metadata for file {f_path} not found. Skipping this file.")

    if not files_to_process:
        raise ValueError("No matching mzML files found for the metadata provided.")

    # Determine the final data ranges and shape.
    # This uses the first file as a template. Be aware that data outside these
    # ranges in other files will be clipped.
    final_rt_range = rt_range
    final_mz_range = mz_range
    final_shape = None
    first_matrix = None

    if final_rt_range is None or final_mz_range is None:
        print("Determining data ranges from the first file...")
        first_matrix, used_rt, used_mz = process_mzml_to_sparse(
            files_to_process[0], rt_precision, mz_precision, mz_range, rt_range
        )
        final_rt_range, final_mz_range = used_rt, used_mz
        final_shape = first_matrix.shape
        print("\nRange and Shape set based on the first file:")
        print(f"  - RT Range: {final_rt_range[0]:.2f} to {final_rt_range[1]:.2f} s")
        print(f"  - m/z Range: {final_mz_range[0]:.4f} to {final_mz_range[1]:.4f}")
        print(f"  - Image Shape: {final_shape}\n")

    print(f"\nProcessing {len(files_to_process)} matched mzML files...")
    
    with h5py.File(save_path, 'w') as f:
        # Create resizable datasets to append data from each file
        dset_data = f.create_dataset('data', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip')
        dset_rt = f.create_dataset('rt_indices', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip')
        dset_mz = f.create_dataset('mz_indices', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip')
        dset_sample = f.create_dataset('sample_indices', shape=(0,), maxshape=(None,), dtype=np.int32, compression='gzip')

        # Main loop to process each file and append its data to the HDF5 datasets
        for i, f_path in enumerate(tqdm(files_to_process, desc="Processing & Writing")):
            if i == 0 and first_matrix is not None:
                sparse_matrix = first_matrix
            else:
                sparse_matrix, _, _ = process_mzml_to_sparse(
                    f_path, rt_precision, mz_precision, final_mz_range, final_rt_range
                )
            
            if final_shape is None: 
                final_shape = sparse_matrix.shape

            intensities = sparse_matrix.data
            rt_indices = sparse_matrix.row
            mz_indices = sparse_matrix.col
            num_points = len(intensities)
            
            # Append data for the current file
            dset_data.resize(dset_data.shape[0] + num_points, axis=0)
            dset_data[-num_points:] = intensities
            
            dset_rt.resize(dset_rt.shape[0] + num_points, axis=0)
            dset_rt[-num_points:] = rt_indices

            dset_mz.resize(dset_mz.shape[0] + num_points, axis=0)
            dset_mz[-num_points:] = mz_indices
            
            dset_sample.resize(dset_sample.shape[0] + num_points, axis=0)
            dset_sample[-num_points:] = [i] * num_points

        print("\nWriting metadata to HDF5 file...")

        # Save the final shape of the 2D matrices
        f.create_dataset('shape', data=final_shape)
        
        # Save all covariates and create mappings for string-based ones
        covariate_keys = list(all_covariates[0].keys())
        all_mappings = {}
        for key in covariate_keys:
            values = [cov[key] for cov in all_covariates]
            if isinstance(values[0], str):
                # For string data, save as byte strings and create an integer mapping
                f.create_dataset(key, data=np.array(values, dtype='S'))
                unique_values = sorted(list(set(values)))
                all_mappings[f"{key}_to_idx"] = {val: i for i, val in enumerate(unique_values)}
            else:
                # For numerical data, save directly
                f.create_dataset(key, data=np.array(values))
        
        # Save the string-to-index mappings as a JSON string in attributes
        if all_mappings:
            f.attrs['mappings'] = json.dumps(all_mappings)

        # Save processing parameters and data ranges as attributes
        f.attrs['rt_precision'] = rt_precision
        f.attrs['mz_precision'] = mz_precision
        f.attrs['rt_range_min'] = final_rt_range[0]
        f.attrs['rt_range_max'] = final_rt_range[1]
        f.attrs['mz_range_min'] = final_mz_range[0]
        f.attrs['mz_range_max'] = final_mz_range[1]
        
    print(f"Done. HDF5 file saved successfully to {save_path}")
