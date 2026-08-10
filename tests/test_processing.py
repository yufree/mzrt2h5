import numpy as np
import pytest
from mzrt2h5 import (
    load_metadata_from_file,
    load_metadata_from_isatab,
    process_mzml_to_sparse,
    save_dataset_as_sparse_h5,
    DynamicSparseH5Dataset,
)

def test_imports():
    assert callable(load_metadata_from_file)
    assert callable(process_mzml_to_sparse)
    assert callable(save_dataset_as_sparse_h5)
    assert callable(DynamicSparseH5Dataset)


# minimal ISA-Tab study + 2 assay files (the MTBLS364 shape: HILIC + RPLC),
# with a non-ASCII unit (µ) — regression for the two bugs that bit MTBLS364:
#   (1) loader read only a_files[0] -> dropped 3/4 of runs;
#   (2) vendor double-extension (.raw.zip) keys didn't match mzML stems.
_S = ("Sample Name\tFactor Value[smoking status]\tCharacteristics[unit]\n"
      "s28\tSmoker\t10 µl\n"
      "s29\tNever Smoker\t10 µl\n")
_A_HILIC = ("Sample Name\tRaw Spectral Data File\n"
            "s28\tHILIC_POS_s28.raw.zip\n"
            "s29\tHILIC_POS_s29.raw.zip\n")
_A_RPLC = ("Sample Name\tRaw Spectral Data File\n"
           "s28\tLipidPOS_s28.raw.zip\n"
           "s29\tLipidPOS_s29.raw.zip\n")


def test_isatab_merges_all_assays_and_keeps_unicode(tmp_path):
    (tmp_path / "s_STUDY.txt").write_text(_S, encoding="utf-8")
    (tmp_path / "a_STUDY_HILIC.txt").write_text(_A_HILIC, encoding="utf-8")
    (tmp_path / "a_STUDY_RPLC.txt").write_text(_A_RPLC, encoding="utf-8")

    m = load_metadata_from_isatab(str(tmp_path))

    # all 4 runs across BOTH assays present (bug 1: only HILIC would appear)
    assert len(m) == 4
    for stem in ("HILIC_POS_s28", "HILIC_POS_s29", "LipidPOS_s28", "LipidPOS_s29"):
        assert stem in m, f"{stem} missing — .raw.zip not stripped or assay dropped"
    # study-level covariate carried per run, unicode intact
    assert m["HILIC_POS_s28"]["smoking status"] == "Smoker"
    assert m["LipidPOS_s29"]["smoking status"] == "Never Smoker"
    assert "µ" in m["HILIC_POS_s28"]["unit"]
