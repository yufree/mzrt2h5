# 0.2.0: CNN classification + interpretability moved to the `mzrtcnn` package
# (model layer). mzrt2h5 is the data layer: mzML→HR H5, simulation, RT alignment,
# and HR→LR rasterization (DynamicSparseH5Dataset, kept here — viz uses it and
# it is a data-layer primitive). `from mzrtcnn import MzrtCNN, train_model, ...`
__version__ = "0.2.0"

from .processing import (load_metadata_from_file, load_metadata_from_mwtab, load_metadata_from_isatab,
                         process_mzml_to_sparse, save_dataset_as_sparse_h5, save_single_mzml_as_sparse_h5,
                         repack_h5)
from .dataset import DynamicSparseH5Dataset
from .visualization import plot_sample_image, plot_ms1ms2_response
from .simulation import generate_simulation_data, simulate_background
from .alignment import compute_rt_corrections, apply_rt_corrections, align_rt
from .acquisition import (download_study, list_study_spectra,
                          download_workbench_study, list_workbench_bundles,
                          fetch_workbench_mwtab, build_h5_from_download)

__all__ = [
    "load_metadata_from_file",
    "load_metadata_from_mwtab",
    "load_metadata_from_isatab",
    "process_mzml_to_sparse",
    "save_dataset_as_sparse_h5",
    "save_single_mzml_as_sparse_h5",
    "DynamicSparseH5Dataset",
    "plot_sample_image",
    "plot_ms1ms2_response",
    "generate_simulation_data",
    "simulate_background",
    "compute_rt_corrections",
    "apply_rt_corrections",
    "align_rt",
    "repack_h5",
    "download_study",
    "list_study_spectra",
    "download_workbench_study",
    "list_workbench_bundles",
    "fetch_workbench_mwtab",
    "build_h5_from_download",
]
