import os
import numpy as np
import pandas as pd
import pymzml
import scipy.sparse as sparse
import h5py
import json
from tqdm import tqdm
from .mzrtsim.reader import SimpleMzMLReader

def analyze_ms1_ms2_response(mzml_path, output_csv_path):
    """
    Analyzes MS1 and MS2 responses (TIC) and cumulative responses over retention time.
    For MS2 scans, it also extracts precursor m/z and product m/z peaks for visualization.
    
    Args:
        mzml_path (str): Path to the mzML file.
        output_csv_path (str): Path to save the result CSV.
    """
    reader = SimpleMzMLReader(mzml_path)
    
    results = []
    
    # Track cumulative sums
    cum_ms1 = 0.0
    cum_ms2 = 0.0
    
    print(f"Analyzing {mzml_path}...")
    
    for spec in reader.get_spectra():
        rt = spec['rt']
        ms_level = spec['ms_level']
        intensity_sum = np.sum(spec['intensity']) if len(spec['intensity']) > 0 else 0.0
        
        if ms_level == 1:
            cum_ms1 += intensity_sum
            current_cum = cum_ms1
        elif ms_level == 2:
            cum_ms2 += intensity_sum
            current_cum = cum_ms2
        else:
            current_cum = 0.0
            
        # Basic scan info
        scan_info = {
            'rt': rt,
            'ms_level': ms_level,
            'tic': intensity_sum,
            'cumulative_tic': current_cum,
            'cum_ms1': cum_ms1,
            'cum_ms2': cum_ms2,
            'precursor_mz': spec.get('precursor_mz'),
            'product_mz': None,
            'product_intensity': None
        }
        
        # If MS2, extract peaks (flattened)
        if ms_level == 2:
            mzs = spec['mz']
            ints = spec['intensity']
            
            # Filter for significant peaks to avoid huge CSVs
            # For visualization, we might just want the whole spectrum or top N
            # Let's take all peaks > 0.1% of base peak or just all if small
            if len(ints) > 0:
                # Simple optimization: only save top 200 peaks if there are many
                if len(ints) > 200:
                    idx = np.argsort(ints)[-200:]
                    mzs = mzs[idx]
                    ints = ints[idx]
                
                for m, i in zip(mzs, ints):
                    # Copy info and add specific peak data
                    row = scan_info.copy()
                    row['product_mz'] = m
                    row['product_intensity'] = i
                    results.append(row)
            else:
                # Keep the scan entry even if empty, just with None
                results.append(scan_info)
        else:
            # For MS1, we don't output every peak (too big), just the TIC summary
            # But the user wants MS1 m/z vs MS2 m/z. 
            # If "MS1 m/z" means "Precursor", then MS1 scans are not the main focus for the scatter plot points,
            # but they drive the cumulative color.
            # So we keep MS1 rows to maintain time continuity if needed, or just skip?
            # Let's keep them but with product_mz = None
            results.append(scan_info)
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Analysis saved to {output_csv_path}")
    return df

def load_metadata_from_file(file_path, sample_id_col='Sample Name', separator=',',
                            format=None):
    """
    Loads metadata from a CSV, TSV, mwTab, or ISA-Tab file into a dictionary.

    Auto-detects format when ``format=None``:
      - Files ending in ``.txt`` containing ``SUBJECT_SAMPLE_FACTORS`` → mwTab
      - Directories or files matching ``s_*.txt`` / ``a_*.txt`` → ISA-Tab
      - Everything else → CSV/TSV (uses ``separator``)

    Args:
        file_path (str): Path to the metadata file or directory (ISA-Tab).
        sample_id_col (str): Column name for sample IDs (CSV/TSV only).
        separator (str): Separator for CSV/TSV files.
        format (str, optional): Force format: ``'csv'``, ``'mwtab'``, or ``'isatab'``.

    Returns:
        dict: A dictionary mapping sample IDs to their metadata.
    """
    if format is None:
        format = _detect_metadata_format(file_path)

    if format == 'mwtab':
        return load_metadata_from_mwtab(file_path)
    elif format == 'isatab':
        return load_metadata_from_isatab(file_path)
    else:
        return _load_metadata_csv(file_path, sample_id_col, separator)


def _detect_metadata_format(file_path):
    """Auto-detect metadata file format."""
    if os.path.isdir(file_path):
        return 'isatab'

    base = os.path.basename(file_path)
    if base.startswith('s_') or base.startswith('a_'):
        return 'isatab'

    if file_path.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('SUBJECT_SAMPLE_FACTORS'):
                        return 'mwtab'
                    if line.startswith('#METABOLOMICS WORKBENCH'):
                        return 'mwtab'
        except Exception:
            pass

    return 'csv'


def _load_metadata_csv(file_path, sample_id_col, separator=','):
    """Load metadata from a CSV or TSV file."""
    try:
        df = pd.read_csv(file_path, sep=separator)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Metadata file not found at {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file with pandas: {e}")

    if sample_id_col not in df.columns:
        raise ValueError(f"Error: The specified sample ID column '{sample_id_col}' was not found in the metadata file.")

    metadata_lookup = df.set_index(sample_id_col).to_dict('index')
    return metadata_lookup


def load_metadata_from_mwtab(file_path):
    """
    Loads metadata from a Metabolomics Workbench mwTab file.

    Parses the SUBJECT_SAMPLE_FACTORS section. Each line has tab-separated
    columns: section_keyword, subject_id, sample_id, factors, [additional_data].
    Factors are pipe-delimited ``key:value`` pairs; additional data uses ``key=value``.

    Args:
        file_path (str): Path to the mwTab file.

    Returns:
        dict: A dictionary mapping sample IDs to their metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"mwTab file not found at {file_path}")

    records = {}
    in_ssf = False

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')

            if line.startswith('SUBJECT_SAMPLE_FACTORS'):
                in_ssf = True
                parts = line.split('\t')
                if len(parts) < 4:
                    continue

                subject_id = parts[1].strip()
                sample_id = parts[2].strip()
                factors_str = parts[3].strip() if len(parts) > 3 else ''
                additional_str = parts[4].strip() if len(parts) > 4 else ''

                record = {}
                if subject_id and subject_id != '-':
                    record['subject_id'] = subject_id

                for pair in factors_str.split('|'):
                    pair = pair.strip()
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        record[k.strip()] = v.strip()

                # Additional data uses key=value separated by | or ;
                if additional_str:
                    for sep in ('|', ';'):
                        if sep in additional_str:
                            break
                    for pair in additional_str.split(sep):
                        pair = pair.strip()
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            record[k.strip()] = v.strip()

                if sample_id:
                    records[sample_id] = record

            elif in_ssf and not line.startswith('SUBJECT_SAMPLE_FACTORS'):
                if line.startswith('#') or line.strip() == '':
                    in_ssf = False

    if not records:
        raise ValueError("No SUBJECT_SAMPLE_FACTORS entries found in the mwTab file.")

    # Re-key by RAW_FILE_NAME (without extension) when available,
    # so entries match mzML filenames on disk. Fall back to sample_id.
    metadata = {}
    for sample_id, record in records.items():
        raw_file = record.pop('RAW_FILE_NAME', None)
        if raw_file:
            file_key = os.path.splitext(os.path.basename(raw_file))[0]
            record['sample_id'] = sample_id
        else:
            file_key = sample_id
        metadata[file_key] = record

    print(f"Loaded {len(metadata)} samples from mwTab file.")
    return metadata


def load_metadata_from_isatab(file_path):
    """
    Loads metadata from MetaboLights ISA-Tab files.

    Accepts either:
      - A directory containing ``s_*.txt`` and ``a_*.txt`` files
      - A single ``s_*.txt`` or ``a_*.txt`` file (the companion is found
        automatically in the same directory)

    The study file provides sample characteristics and factor values.
    The assay file provides the mapping from sample names to raw data filenames
    (the ``Raw Spectral Data File`` column).

    The returned dictionary is keyed by mzML filename (without extension),
    so it can be matched directly against mzML filenames on disk.

    Args:
        file_path (str): Path to ISA-Tab directory or an ``s_*/a_*`` file.

    Returns:
        dict: A dictionary mapping mzML filenames (no extension) to metadata.
    """
    if os.path.isdir(file_path):
        search_dir = file_path
    else:
        search_dir = os.path.dirname(file_path)

    s_files = sorted([f for f in os.listdir(search_dir) if f.startswith('s_') and f.endswith('.txt')])
    a_files = sorted([f for f in os.listdir(search_dir) if f.startswith('a_') and f.endswith('.txt')])

    if not s_files and not a_files:
        raise FileNotFoundError(f"No ISA-Tab study (s_*.txt) or assay (a_*.txt) files found in {search_dir}")

    study_df = None
    if s_files:
        study_df = pd.read_csv(os.path.join(search_dir, s_files[0]), sep='\t')

    assay_df = None
    if a_files:
        assay_df = pd.read_csv(os.path.join(search_dir, a_files[0]), sep='\t')

    # Extract useful columns from study file
    study_meta = {}
    if study_df is not None and 'Sample Name' in study_df.columns:
        useful_cols = [c for c in study_df.columns
                       if c.startswith('Characteristics[') or c.startswith('Factor Value[')
                       or c in ('Source Name', 'Sample Name', 'Material Type')]
        for _, row in study_df[useful_cols].iterrows():
            sample_name = str(row['Sample Name'])
            record = {}
            for col in useful_cols:
                if col == 'Sample Name':
                    continue
                val = row[col]
                if pd.notna(val):
                    clean_col = col.replace('Characteristics[', '').replace('Factor Value[', '').rstrip(']')
                    record[clean_col] = str(val)
            study_meta[sample_name] = record

    # Build mapping from sample name to raw filename using assay file
    sample_to_file = {}
    if assay_df is not None:
        raw_col = None
        for c in assay_df.columns:
            if 'raw' in c.lower() and 'file' in c.lower():
                raw_col = c
                break
        if raw_col and 'Sample Name' in assay_df.columns:
            for _, row in assay_df.iterrows():
                sample_name = str(row['Sample Name'])
                raw_file = str(row[raw_col]) if pd.notna(row[raw_col]) else ''
                if raw_file:
                    sample_to_file[sample_name] = raw_file

            # Also extract assay-level metadata
            assay_cols = [c for c in assay_df.columns
                          if c.startswith('Parameter Value[') or c == 'MS Assay Name']
            if assay_cols:
                for _, row in assay_df.iterrows():
                    sample_name = str(row['Sample Name'])
                    if sample_name in study_meta:
                        for col in assay_cols:
                            val = row[col]
                            if pd.notna(val):
                                clean_col = col.replace('Parameter Value[', '').rstrip(']')
                                study_meta[sample_name][clean_col] = str(val)

    # Re-key by mzML filename (without extension) for matching
    metadata = {}
    if sample_to_file:
        for sample_name, raw_file in sample_to_file.items():
            file_key = os.path.basename(raw_file)
            file_key = os.path.splitext(file_key)[0]
            record = study_meta.get(sample_name, {})
            record['sample_name'] = sample_name
            metadata[file_key] = record
    else:
        # No assay file: key directly by sample name (user will match by filename)
        metadata = study_meta

    if not metadata:
        raise ValueError("No sample metadata could be extracted from the ISA-Tab files.")

    print(f"Loaded {len(metadata)} samples from ISA-Tab files.")
    return metadata
    
def process_mzml_to_sparse(file, rt_precision, mz_precision, mz_range=None, rt_range=None, min_rel_intensity=None):
    """
    Processes a single mzML file into a sparse 2D matrix (RT vs. m/z).

    Args:
        file (str): Path to the .mzML file.
        rt_precision (float): The bin size for the retention time axis.
        mz_precision (float): The bin size for the m/z axis.
        mz_range (tuple, optional): A (min, max) tuple to fix the m/z range.
        rt_range (tuple, optional): A (min, max) tuple to fix the RT range.
        min_rel_intensity (float, optional): Keep only points >= this fraction of scan base peak.

    Returns:
        tuple: A COO sparse matrix, the used RT range, and the used m/z range.
    """
    run = pymzml.run.Reader(file)
    
    spectra_data = []
    for spectrum in run:
        # Process only MS1 level scans with actual data points
        if spectrum.ms_level == 1 and len(spectrum.mz) > 0:
            intensities = spectrum.i.astype(np.float32)
            mz = spectrum.mz
            
            if min_rel_intensity is not None:
                max_i = np.max(intensities)
                if max_i > 0:
                    mask = (intensities / max_i) >= min_rel_intensity
                    mz = mz[mask]
                    intensities = intensities[mask]
            
            if len(mz) > 0:
                spectra_data.append({
                    "rt": spectrum.scan_time_in_minutes() * 60, # Convert RT to seconds
                    "mz": mz,
                    "intensity": intensities.astype(np.int32)
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
                              format=None,
                              mz_range=None, rt_range=None,
                              min_rel_intensity=None,
                              progress_callback=None):
    """
    Processes a folder of mzML files and saves them as a single, consolidated
    sparse HDF5 file with associated metadata.

    Args:
        folder (str): Path to the folder containing .mzML files.
        save_path (str): Path to save the output .h5 file.
        rt_precision (float): Bin size for the retention time axis.
        mz_precision (float): Bin size for the m/z axis.
        metadata_csv_path (str): Path to the metadata file (CSV, mwTab, or ISA-Tab dir).
        sample_id_col (str): Column name for sample IDs (CSV/TSV only).
        separator (str): Separator for CSV/TSV metadata files.
        format (str, optional): Metadata format: 'csv', 'mwtab', 'isatab', or None (auto).
        mz_range (tuple, optional): Fixed (min, max) m/z range.
        rt_range (tuple, optional): Fixed (min, max) RT range.
        min_rel_intensity (float, optional): Keep only points >= this fraction of scan base peak.
        progress_callback (function, optional): Callback function to report progress updates.
    """

    # Load metadata from the provided file
    metadata_lookup = load_metadata_from_file(metadata_csv_path, sample_id_col, separator, format=format)
    print(f"Successfully loaded {len(metadata_lookup)} metadata records from {metadata_csv_path}.")
    if progress_callback:
        progress_callback({'step': 'loading_metadata', 'status': 'completed', 'message': 'Loaded metadata', 'progress': 10})

    files_to_process = []
    all_covariates = []
    
    # Find all .mzML files and match them with loaded metadata
    all_mzml_files = sorted([os.path.join(root, f) for root, _, fs in os.walk(folder) for f in fs if f.endswith('.mzML')])
    
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

    if progress_callback:
        progress_callback({'step': 'matching_files', 'status': 'completed', 'message': f'Matched {len(files_to_process)} files', 'progress': 20})

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
            files_to_process[0], rt_precision, mz_precision, mz_range, rt_range, min_rel_intensity
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
        # lzf decompresses 5–10× faster than gzip; chunk size of 100 000 avoids
        # the ~50 000 decompress calls that the default chunk size of 1024 would need.
        _h5_kw = dict(dtype=np.int32, compression='lzf', chunks=(100000,))
        dset_data = f.create_dataset('data', shape=(0,), maxshape=(None,), **_h5_kw)
        dset_rt = f.create_dataset('rt_indices', shape=(0,), maxshape=(None,), **_h5_kw)
        dset_mz = f.create_dataset('mz_indices', shape=(0,), maxshape=(None,), **_h5_kw)
        dset_sample = f.create_dataset('sample_indices', shape=(0,), maxshape=(None,), **_h5_kw)

        if progress_callback:
            progress_callback({'step': 'initializing_hdf5', 'status': 'completed', 'message': 'Initialized HDF5 file', 'progress': 30})

        # Main loop to process each file and append its data to the HDF5 datasets
        total_files = len(files_to_process)
        written_indices = []  # indices into files_to_process of files actually written
        for i, f_path in enumerate(tqdm(files_to_process, desc="Processing & Writing")):
            if progress_callback:
                progress = 30 + int((i / total_files) * 50)
                progress_callback({
                    'step': 'processing_files',
                    'status': 'in_progress',
                    'message': f'Processing file {i+1}/{total_files}: {os.path.basename(f_path)}',
                    'progress': progress,
                    'file_index': i+1,
                    'total_files': total_files
                })

            if i == 0 and first_matrix is not None:
                sparse_matrix = first_matrix
            else:
                sparse_matrix, _, _ = process_mzml_to_sparse(
                    f_path, rt_precision, mz_precision, final_mz_range, final_rt_range, min_rel_intensity
                )

            if final_shape is None:
                final_shape = sparse_matrix.shape

            intensities = sparse_matrix.data
            rt_indices = sparse_matrix.row
            mz_indices = sparse_matrix.col
            num_points = len(intensities)

            # Skip files with no data points (e.g., blank samples)
            if num_points == 0:
                print(f"  Skipping {os.path.basename(f_path)} - no data points.")
                continue

            # Use the count of already-written samples as the sample index to avoid gaps
            written_sample_idx = len(written_indices)
            written_indices.append(i)

            # Append data for the current file
            dset_data.resize(dset_data.shape[0] + num_points, axis=0)
            dset_data[-num_points:] = intensities

            dset_rt.resize(dset_rt.shape[0] + num_points, axis=0)
            dset_rt[-num_points:] = rt_indices

            dset_mz.resize(dset_mz.shape[0] + num_points, axis=0)
            dset_mz[-num_points:] = mz_indices

            dset_sample.resize(dset_sample.shape[0] + num_points, axis=0)
            dset_sample[-num_points:] = [written_sample_idx] * num_points

        print("\nWriting metadata to HDF5 file...")

        # Filter covariates and file list to only include files that were actually written
        written_covariates = [all_covariates[i] for i in written_indices]
        written_files = [files_to_process[i] for i in written_indices]

        # Save the final shape of the 2D matrices
        f.create_dataset('shape', data=final_shape)

        # Save all covariates and create mappings for string-based ones
        covariate_keys = list(written_covariates[0].keys()) if written_covariates else []
        all_mappings = {}
        for key in covariate_keys:
            values = [cov[key] for cov in written_covariates]
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

        # Store sample IDs (filenames without extension) for downstream workflows
        sample_ids = [os.path.basename(fp).replace('.mzML', '') for fp in written_files]
        f.create_dataset('sample_id', data=np.array(sample_ids, dtype='S'))

        # Save processing parameters and data ranges as attributes
        f.attrs['rt_precision'] = rt_precision
        f.attrs['mz_precision'] = mz_precision
        f.attrs['rt_range_min'] = final_rt_range[0]
        f.attrs['rt_range_max'] = final_rt_range[1]
        f.attrs['mz_range_min'] = final_mz_range[0]
        f.attrs['mz_range_max'] = final_mz_range[1]
        
    if progress_callback:
        progress_callback({'step': 'writing_hdf5', 'status': 'completed', 'message': 'Writing to HDF5 file completed', 'progress': 90})
        progress_callback({'step': 'completed', 'status': 'completed', 'message': 'Processing completed', 'progress': 100})
    print(f"Done. HDF5 file saved successfully to {save_path}")

def save_single_mzml_as_sparse_h5(mzml_file_path, save_path, rt_precision, mz_precision,
                                   mz_range=None, rt_range=None, sample_name=None,
                                   min_rel_intensity=None,
                                   progress_callback=None):
    """
    Processes a single mzML file and saves it as a sparse HDF5 file.

    Args:
        mzml_file_path (str): Path to the .mzML file to process.
        save_path (str): Path to save the output .h5 file.
        rt_precision (float): Bin size for the retention time axis.
        mz_precision (float): Bin size for the m/z axis.
        mz_range (tuple, optional): Fixed (min, max) m/z range.
        rt_range (tuple, optional): Fixed (min, max) RT range.
        sample_name (str, optional): Name to use for the sample. If None, uses the filename.
        progress_callback (function, optional): Callback function to report progress updates.
    """
    if not os.path.exists(mzml_file_path):
        raise FileNotFoundError(f"mzML file not found at {mzml_file_path}")

    if sample_name is None:
        sample_name = os.path.basename(mzml_file_path).replace('.mzML', '')

    print(f"Processing single mzML file: {mzml_file_path}")
    print(f"Sample name: {sample_name}")
    if progress_callback:
        progress_callback({'step': 'initializing', 'status': 'in_progress', 'message': 'Processing single file', 'progress': 10})

    sparse_matrix, used_rt_range, used_mz_range = process_mzml_to_sparse(
        mzml_file_path, rt_precision, mz_precision, mz_range, rt_range, min_rel_intensity
    )
    if progress_callback:
        progress_callback({'step': 'processing_file', 'status': 'completed', 'message': 'Processed file data', 'progress': 50})

    final_rt_range = rt_range if rt_range is not None else used_rt_range
    final_mz_range = mz_range if mz_range is not None else used_mz_range
    final_shape = sparse_matrix.shape

    print(f"\nData ranges and shape:")
    print(f"  - RT Range: {final_rt_range[0]:.2f} to {final_rt_range[1]:.2f} s")
    print(f"  - m/z Range: {final_mz_range[0]:.4f} to {final_mz_range[1]:.4f}")
    print(f"  - Matrix Shape: {final_shape}\n")

    try:
        with h5py.File(save_path, 'w') as f:
            if progress_callback:
                progress_callback({'step': 'writing_hdf5', 'status': 'in_progress', 'message': 'Writing to HDF5 file', 'progress': 70})
            
            intensities = sparse_matrix.data
            rt_indices = sparse_matrix.row
            mz_indices = sparse_matrix.col

            _h5_kw = dict(compression='lzf', chunks=(100000,))
            f.create_dataset('data', data=intensities, **_h5_kw)
            f.create_dataset('rt_indices', data=rt_indices, **_h5_kw)
            f.create_dataset('mz_indices', data=mz_indices, **_h5_kw)
            f.create_dataset('sample_indices', data=np.zeros(len(intensities), dtype=np.int32), **_h5_kw)
            f.create_dataset('sample_name', data=np.array([sample_name], dtype='S'))
            f.create_dataset('shape', data=final_shape)

            f.attrs['rt_precision'] = rt_precision
            f.attrs['mz_precision'] = mz_precision
            f.attrs['rt_range_min'] = final_rt_range[0]
            f.attrs['rt_range_max'] = final_rt_range[1]
            f.attrs['mz_range_min'] = final_mz_range[0]
            f.attrs['mz_range_max'] = final_mz_range[1]
            
    except Exception as e:
        print(f"ERROR writing HDF5 file: {e}")
        raise

    # Only report completion if the file was actually created
    if os.path.exists(save_path):
        if progress_callback:
            progress_callback({'step': 'completed', 'status': 'completed', 'message': 'Processing completed', 'progress': 100})
        print(f"Done. HDF5 file saved successfully to {save_path}")
    else:
        if progress_callback:
            progress_callback({'step': 'error', 'status': 'error', 'message': 'Failed to create HDF5 file', 'progress': -1})
        print(f"ERROR: HDF5 file was not created at {save_path}")


def repack_h5(input_path, output_path=None, compression='lzf', chunk_size=100000,
              copy_chunk=10_000_000):
    """Repack an existing HDF5 file with a new compression codec and chunk size.

    Typical use: convert an older gzip-compressed file to lzf for faster reads
    (lzf decompresses 3–5× faster than gzip).

    Args:
        input_path:  Path to the source HDF5 file.
        output_path: Destination path (default: input filename with ``_repacked`` suffix).
        compression: Compression codec: ``'lzf'`` (fast reads), ``'gzip'`` (small
                     files), or ``None`` (no compression).
        chunk_size:  HDF5 dataset chunk size in number of elements.
        copy_chunk:  Number of data points copied per iteration.

    Returns:
        str: Path to the repacked output file.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_repacked{ext}"

    kw = dict(compression=compression, chunks=(chunk_size,))
    data_names = ['data', 'rt_indices', 'mz_indices', 'sample_indices']

    with h5py.File(input_path, 'r') as fin, h5py.File(output_path, 'w') as fout:
        total = len(fin['data'])
        for name in data_names:
            fout.create_dataset(name, shape=(total,), dtype=fin[name].dtype, **kw)

        for start in tqdm(range(0, total, copy_chunk), desc="Repacking",
                          total=(total + copy_chunk - 1) // copy_chunk):
            end = min(start + copy_chunk, total)
            for name in data_names:
                fout[name][start:end] = fin[name][start:end]

        for name in fin.keys():
            if name not in data_names:
                fin.copy(name, fout)
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

    in_size = os.path.getsize(input_path) / 1e9
    out_size = os.path.getsize(output_path) / 1e9
    print(f"Repacked: {in_size:.2f} GB → {out_size:.2f} GB ({compression})")
    return output_path
