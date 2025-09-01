# mzrt2h5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package to convert mzML files to HDF5 format for deep learning applications.

## Installation

After installation, a new command `mzrt2h5` will be available in your terminal.

```bash
pip install -r requirements.txt
pip install .
```

## CLI Usage

This is the most straightforward way to use the package. After installation, you can call the `mzrt2h5` command from your terminal.

**Example:**

```bash
mzrt2h5 \
    /path/to/your/mzml_folder/ \
    /path/to/your/output.h5 \
    --metadata-csv-path /path/to/your/metadata.csv \
    --rt-precision 0.1 \
    --mz-precision 0.01
```

**Options:**

Use `mzrt2h5 --help` to see all available options.

## Python Usage

```python
from mzrt2h5.processing import save_dataset_as_sparse_h5
from mzrt2h5.dataset import DynamicSparseH5Dataset

# Process mzML files and save to HDF5
save_dataset_as_sparse_h5(
    folder="path/to/your/mzML_files",
    save_path="output.h5",
    rt_precision=0.1,
    mz_precision=0.01,
    metadata_csv_path="path/to/your/metadata.csv",
)

# Create a PyTorch dataset
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
```

## Web Interface

This package also includes a simple web interface to perform the conversion.

1.  **Run the Flask app:**
    ```bash
    python app/app.py
    ```

2.  **Access the web interface:**
    Open your web browser and go to `http://127.0.0.1:5000`.

3.  **Use the form:**
    Upload your metadata file, select your mzML files directory, set the parameters, and click "Process and Download H5".
