# mzrt2h5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package to convert mzML mass spectrometry files to HDF5 format for deep learning. Supports metadata from CSV, Metabolomics Workbench (mwTab), and MetaboLights (ISA-Tab). Version 0.2.0

## Design Philosophy

`mzrt2h5` stores data at the **highest practical resolution** (default 0.0001 Da, 0.1 s) in a sparse HDF5 format. This high-resolution (HR) storage is the data foundation for the downstream packages of the `mzrt*` suite:

- **`mzrtpeak`** — streaming high-resolution peak picking (0.01 Da EICs + 1D `find_peaks`, no training), giving ppm-level mass accuracy.
- **`mzrtcnn`** — classifies samples on a low-resolution image (downsampled here via `DynamicSparseH5Dataset`) and uses attribution to flag the regions driving a prediction; `mzrtpeak` then resolves the peaks inside those regions.
- **`mzrtgnn`** — *(in development)* consumes the peak table for PMD-based metabolic network analysis.

The sparse format means that increasing m/z resolution from 0.01 Da to 0.0001 Da adds no storage overhead beyond the actual non-zero data points already present in the mzML files.

## Installation

```bash
pip install mzrt2h5
```

## Quick Start

Put your mzML files in a folder, point the tool at your metadata file, and get an HDF5 file ready for deep learning:

```bash
# CSV metadata
mzrt2h5 process ./mzml_folder/ output.h5 --metadata-csv-path metadata.csv

# Metabolomics Workbench mwTab (auto-detected)
mzrt2h5 process ./mzml_folder/ output.h5 --metadata-csv-path ST000001.txt

# MetaboLights ISA-Tab (auto-detected from directory)
mzrt2h5 process ./mzml_folder/ output.h5 --metadata-csv-path ./MTBLS123/
```

The format is auto-detected. You can also force it with `--metadata-format`:

```bash
mzrt2h5 process ./mzml_folder/ output.h5 \
    --metadata-csv-path ST000001.txt \
    --metadata-format mwtab
```

## Metadata Formats

### CSV / TSV

A plain table where one column contains sample IDs that match your mzML filenames (without the `.mzML` extension).

| Sample Name | class  | batch |
|-------------|--------|-------|
| sample_01   | control| 1     |
| sample_02   | treated| 1     |

```bash
mzrt2h5 process ./mzml/ output.h5 \
    --metadata-csv-path metadata.csv \
    --sample-id-col "Sample Name" \
    --separator ","
```

### Metabolomics Workbench (mwTab)

Download the mwTab file from Metabolomics Workbench (e.g., `ST000001.txt`). The parser reads the `SUBJECT_SAMPLE_FACTORS` section, which encodes factors as `key:value | key:value` and additional data as `key=value;key=value`.

If the additional data contains a `RAW_FILE_NAME` field, it is used to match mzML filenames automatically. Otherwise, sample IDs are matched against filenames directly.

```
SUBJECT_SAMPLE_FACTORS	SU001	Sample_A	Treatment:Control | Gender:Male	RAW_FILE_NAME=sample_a.mzML
SUBJECT_SAMPLE_FACTORS	SU002	Sample_B	Treatment:Disease | Gender:Female	RAW_FILE_NAME=sample_b.mzML
```

```bash
mzrt2h5 process ./mzml/ output.h5 --metadata-csv-path ST000001.txt
```

### MetaboLights (ISA-Tab)

Download the study metadata from MetaboLights. Point the tool at the directory containing `s_*.txt` (study) and `a_*.txt` (assay) files, or at a single file (the companion is found automatically).

- The study file (`s_*.txt`) provides `Characteristics[*]` and `Factor Value[*]` columns.
- The assay file (`a_*.txt`) maps `Sample Name` to `Raw Spectral Data File` (your mzML filenames).

```bash
# Point at the ISA-Tab directory
mzrt2h5 process ./mzml/ output.h5 --metadata-csv-path ./MTBLS123/

# Or point at a single file
mzrt2h5 process ./mzml/ output.h5 --metadata-csv-path ./MTBLS123/s_MTBLS123.txt
```

## CLI Reference

### `mzrt2h5 process` — Batch Processing

```
mzrt2h5 process FOLDER SAVE_PATH [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--metadata-csv-path` | (required) | Path to metadata file (CSV, mwTab, or ISA-Tab directory) |
| `--metadata-format` | `auto` | Force format: `auto`, `csv`, `mwtab`, `isatab` |
| `--sample-id-col` | `Sample Name` | Column name for sample IDs (CSV/TSV only) |
| `--separator` | `,` | Separator for CSV/TSV files |
| `--rt-precision` | `0.1` | Bin size for the retention time axis (seconds) |
| `--mz-precision` | `0.0001` | Bin size for the m/z axis (Da). 0.0001 Da gives <0.5 ppm at m/z 500 for HR restoration |
| `--mz-range` | auto | Fixed (min, max) m/z range |
| `--rt-range` | auto | Fixed (min, max) RT range |

### `mzrt2h5 plot` — Visualize a Sample

```bash
mzrt2h5 plot output.h5 "Sample_A" --rt-precision 0.5 --mz-precision 0.05 --output-path plot.png
```

### `mzrt2h5 ms1ms2` — MS1/MS2 TIC Analysis

```bash
mzrt2h5 ms1ms2 input.mzML output.csv
```

## Python API

### Batch Processing

```python
from mzrt2h5 import save_dataset_as_sparse_h5

# CSV metadata
save_dataset_as_sparse_h5(
    folder="path/to/mzML_files",
    save_path="output.h5",
    rt_precision=0.1,
    mz_precision=0.0001,
    metadata_csv_path="metadata.csv",
)

# mwTab (auto-detected)
save_dataset_as_sparse_h5(
    folder="path/to/mzML_files",
    save_path="output.h5",
    rt_precision=0.1,
    mz_precision=0.0001,
    metadata_csv_path="ST000001.txt",
)

# ISA-Tab (auto-detected)
save_dataset_as_sparse_h5(
    folder="path/to/mzML_files",
    save_path="output.h5",
    rt_precision=0.1,
    mz_precision=0.0001,
    metadata_csv_path="./MTBLS123/",
)
```

### Metadata Parsers (Standalone)

```python
from mzrt2h5 import load_metadata_from_file, load_metadata_from_mwtab, load_metadata_from_isatab

# Auto-detect format
meta = load_metadata_from_file("ST000001.txt")
meta = load_metadata_from_file("./MTBLS123/")
meta = load_metadata_from_file("metadata.csv")

# Or call parsers directly
meta = load_metadata_from_mwtab("ST000001.txt")
meta = load_metadata_from_isatab("./MTBLS123/")
```

All parsers return `dict[str, dict[str, str]]` — a mapping from sample ID (or mzML filename without extension) to a dictionary of covariates.

### Single File Conversion

```python
from mzrt2h5 import save_single_mzml_as_sparse_h5

save_single_mzml_as_sparse_h5(
    mzml_file_path="sample.mzML",
    save_path="output.h5",
    rt_precision=0.1,
    mz_precision=0.0001,
)
```

### PyTorch Dataset

```python
from mzrt2h5 import DynamicSparseH5Dataset

dataset = DynamicSparseH5Dataset(
    h5_path="output.h5",
    target_rt_precision=0.5,
    target_mz_precision=0.05,
)

# With data augmentation for training
train_dataset = DynamicSparseH5Dataset(
    h5_path="output.h5",
    target_rt_precision=0.5,
    target_mz_precision=0.05,
    augment=True,
    aug_rt_shift_s=30,
    aug_mz_shift_ppm=5,
)
```

### CNN Classification

> CNN classification and interpretability live in the separate **`mzrtcnn`**
> package (the model layer). `mzrt2h5` is the data layer: it provides the
> `DynamicSparseH5Dataset` that downsamples the HR H5 to a low-resolution image,
> which `mzrtcnn` consumes for training and attribution.

```python
from mzrtcnn import MzrtCNN, train_model, predict  # pip install mzrtcnn

result = train_model("output.h5", target_covariate="class", num_epochs=50)
preds = predict("new_data.h5", model_path="model.pth")
```

### RT Alignment

```python
from mzrt2h5 import align_rt

# One-stop: compute QC-aware corrections and apply them.
# Writes rt_aligned=True to the H5 attributes so downstream tools skip re-alignment.
align_rt("output.h5", output_path="aligned.h5")

# Or in two steps (e.g. to inspect/reuse corrections):
from mzrt2h5 import compute_rt_corrections, apply_rt_corrections

corrections = compute_rt_corrections("output.h5")
apply_rt_corrections("output.h5", corrections, output_path="aligned.h5")
```

### Visualization

```python
from mzrt2h5 import plot_sample_image

plot_sample_image(
    h5_path="output.h5",
    sample_id="Sample_A",
    target_rt_precision=0.5,
    target_mz_precision=0.05,
    output_path="plot.png",
)
```

## Web Interface

```bash
python app/app.py
```

Open `http://127.0.0.1:5002` to access the web interface with real-time progress tracking.

## Changelog

### Version 0.2.0
- **Moved** the CNN classification model and trainer (`MzrtCNN`, `train_model`, `predict`, `cross_validate`) out of `mzrt2h5` into the dedicated **`mzrtcnn`** package (the model layer). `mzrt2h5` is now strictly the data layer: mzML→HR sparse H5, simulation, RT alignment, and HR→LR rasterization (`DynamicSparseH5Dataset`, kept here as a data-layer primitive used by both viz and `mzrtcnn`).
- **Removed** `mzrt2h5.model` and `mzrt2h5.trainer`. Install `mzrtcnn` and use `from mzrtcnn import MzrtCNN, train_model, predict`.

### Version 0.1.10
- **Added** sharp chemical-noise simulation: `noise_peaks`, `noise_peak_sigma`, `noise_peak_snr` in `simmzml()` and `generate_simulation_data()`. Injects narrow, isolated, peak-shaped noise events into matrix (pure-noise) m/z channels (requires `matrix=True`). Flat Gaussian baseline alone is unrealistic and overstates how well shape/SNR features separate real peaks from noise; this reproduces real chemical noise so peak-quality scoring can be validated honestly.

### Version 0.1.9
- **Added** metadata support for Metabolomics Workbench mwTab format (`load_metadata_from_mwtab`).
- **Added** metadata support for MetaboLights ISA-Tab format (`load_metadata_from_isatab`).
- **Added** auto-detection of metadata format in `load_metadata_from_file` and CLI (`--metadata-format`).
- **Added** `--metadata-format` option to CLI `process` command (`auto`/`csv`/`mwtab`/`isatab`).
- **Added** `repack_h5()` to repack an existing HDF5 file with a different compression codec (e.g. gzip → lzf) and chunk size for faster downstream reads.
- **Added** `min_rel_intensity` parameter to `process_mzml_to_sparse()` and `save_dataset_as_sparse_h5()` to filter low-intensity peaks below a fraction of each scan's base peak.
- **Rewritten** RT alignment module: replaced the single `align_h5()` function with a three-function API — `compute_rt_corrections()`, `apply_rt_corrections()`, and the convenience wrapper `align_rt()` — adding QC-aware reference selection, segmented BPC cross-correlation with spline smoothing, and a streaming two-pass strategy that keeps memory under ~10 MB regardless of file size.
- **Fixed** version mismatch between `__init__.py` and `pyproject.toml`.
- **Fixed** Flask app `simulate` endpoint argument order and missing `jsonify` import.
- **Fixed** path traversal vulnerability in Flask `download_file` endpoint.
- **Fixed** bare `except` clause in `DynamicSparseH5Dataset`.
- **Fixed** silent exception swallowing in `save_single_mzml_as_sparse_h5`.

### Version 0.1.8
- **Added** CNN end-to-end deep learning model (`MzrtCNN`) for sample classification directly on sparse 2D mass spec images via `mzrt2h5.model` and `mzrt2h5.trainer`.
- **Added** RT alignment module (`align_h5`) using base peak chromatogram (BPC) cross-correlation via `mzrt2h5.alignment`.
- **Enhanced** `DynamicSparseH5Dataset` by supporting `target_covariate` for classification tasks.

### Version 0.1.7
- **Added** simulated intensity column (`sim_ins`) to CSV output of mzML simulation.

### Version 0.1.6
- **Enhanced** simulation capabilities: `pwidth`, `snr`, `rtime` accept per-compound vectors; `baseline` accepts time-varying vector; `tailingindex` selects tailing compounds.
- **Fixed** `DynamicSparseH5Dataset` handling of empty spectra samples.

### Version 0.1.5
- **Added** 0-compound simulation and `mzrtsim` module.

### Version 0.1.4
- **Fixed** path resolution, error handling, progress tracking in web interface.
