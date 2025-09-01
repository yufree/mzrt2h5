import pytest
from mzrt2h5 import (
    load_metadata_from_file,
    process_mzml_to_sparse,
    save_dataset_as_sparse_h5,
    DynamicSparseH5Dataset,
)

def test_imports():
    assert callable(load_metadata_from_file)
    assert callable(process_mzml_to_sparse)
    assert callable(save_dataset_as_sparse_h5)
    assert callable(DynamicSparseH5Dataset)
