# mzrt2h5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package to convert mzML files to HDF5 format for deep learning applications. Version 0.1.7

## Installation

```bash
pip install mzrt2h5
```

After installation, a new command `mzrt2h5` will be available in your terminal.

## CLI Usage

This is the most straightforward way to use the package. After installation, you can call the `mzrt2h5` command from your terminal.

### Batch Processing (Multiple Files)

**Example:**

```bash
mzrt2h5 process \
    /path/to/your/mzml_folder/ \
    /path/to/your/output.h5 \
    --metadata-csv-path /path/to/your/metadata.csv \
    --rt-precision 0.1 \
    --mz-precision 0.01
```

### Single File Conversion

To convert a single mzML file without needing metadata:

**Example:**

```bash
mzrt2h5 process-single \
    /path/to/your/file.mzML \
    /path/to/your/output.h5 \
    --rt-precision 0.1 \
    --mz-precision 0.01
```

**Options:**

Use `mzrt2h5 --help` to see all available options.

## Python Usage

### Batch Processing (Multiple Files)

```python
from mzrt2h5.processing import save_dataset_as_sparse_h5
from mzrt2h5.dataset import DynamicSparseH5Dataset
from mzrt2h5.visualization import plot_sample_image

# Process mzML files and save to HDF5
save_dataset_as_sparse_h5(
    folder="path/to/your/mzML_files",
    save_path="output.h5",
    rt_precision=0.1,
    mz_precision=0.01,
    metadata_csv_path="path/to/your/metadata.csv",
)
```

### Single File Conversion

```python
from mzrt2h5.processing import save_single_mzml_as_sparse_h5

# Process a single mzML file and save to HDF5
save_single_mzml_as_sparse_h5(
    mzml_file_path="path/to/your/file.mzML",
    save_path="output.h5",
    rt_precision=0.1,
    mz_precision=0.01,
)
```

# Create a PyTorch dataset

```python
dataset = DynamicSparseH5Dataset(
    h5_path="output.h5",
    target_rt_precision=0.5,
    target_mz_precision=0.05,
)

# Create a dataset with on-the-fly augmentation for training
# with a random retention time shift of +/- 30 seconds
# and a random m/z shift of +/- 5 ppm.
train_dataset = DynamicSparseH5Dataset(
    h5_path="output.h5",
    target_rt_precision=0.5,
    target_mz_precision=0.05,
    augment=True,
    aug_rt_shift_s=30,
    aug_mz_shift_ppm=5
)

# Plot a sample image from the HDF5 file
plot_sample_image(
    h5_path="output.h5",
    sample_id="Sample_A", # Or an integer index like 0
    target_rt_precision=0.5,
    target_mz_precision=0.05,
    output_path="sample_A_plot.png" # Saves to file, remove to display interactively
)
```

## Visualization

To visualize a mass spectrometry image from your HDF5 file, use the `mzrt2h5 plot` command:

```bash
mzrt2h5 plot \
    /path/to/your/output.h5 \
    "Sample_A" \
    --rt-precision 0.5 \
    --mz-precision 0.05 \
    --output-path sample_A_plot.png
```

**Options:**

Use `mzrt2h5 plot --help` to see all available options for plotting.

## Changelog

### Version 0.1.7

- **Added** simulated intensity column (`sim_ins`) to CSV output of mzML simulation:
  - The new column shows the maximum simulated intensity (peak height) for each compound peak.
  - Values match the theoretical maximum that peak detection algorithms should find.
  - Supports both `simmzml` and `simmzml_background` simulation functions.
  - Useful for validating peak finding algorithms and understanding simulation parameters.

### Version 0.1.6

- **Enhanced** simulation capabilities in `generate_simulation_data`:
  - `pwidth`, `snr`, and `rtime` now accept vectors to specify values per compound.
  - `baseline` accepts a vector to simulate baseline shifts over time.
  - `tailingindex` allows specifying which compounds exhibit tailing.
- **Fixed** `DynamicSparseH5Dataset` to correctly handle samples with no peaks (empty spectra), ensuring robust loading and label handling.

### Version 0.1.5

- **Added** support for 0-compound simulation in `mzrtsim` to enable matrix-only simulations, useful for generating blank matrix data.
- **Added** support for `mzrtsim` for mzml simulation.

### Version 0.1.4

- **Fixed** path resolution issues in the web interface to ensure HDF5 files are properly located
- **Improved** error handling in HDF5 file writing
- **Updated** default precision values in the web interface (rt_precision: 1.0, mz_precision: 0.001)
- **Enhanced** progress tracking and debugging in both CLI and web interface
- **Added** better file extension handling for output filenames
- **Fixed** version consistency across all package files

## Web Interface

This package includes a web interface with real-time progress indicators for both single-file and batch processing.

1.  **Run the Flask app:**
    ```bash
    python app/app.py
    ```

2.  **Access the web interface:**
    Open your web browser and go to `http://127.0.0.1:5002`.

3.  **Use the interface:**
    The web interface has two modes:
    - **Batch Processing**: Upload a metadata file and multiple mzML files for processing
    - **Single File**: Upload a single mzML file without needing metadata

    Select the appropriate tab, set the parameters, and click the "Process" button.

4.  **Monitor progress:**
    - Real-time progress bar shows processing status from 0% to 100%
    - Detailed status messages indicate current processing stage
    - Progress updates automatically without page refresh

5.  **Download results:**
    - Download button appears automatically when processing completes
    - Click to download the generated HDF5 file
    - Temporary files are automatically cleaned up after download
