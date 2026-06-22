import os
import pytest
import pandas as pd
from mzrt2h5.mzrtsim import load_data, simmzml

def test_zero_compounds_simulation(tmp_path):
    # Use tmp_path fixture from pytest to avoid creating files in project root
    output_dir = tmp_path / "sim_output"
    output_dir.mkdir()
    output_name = str(output_dir / "test_zero")
    
    db = load_data('monahrms1')
    
    # Run simulation with 0 compounds and matrix=True
    # We need to suppress the print output or just let it be
    # Also matrix=True requires mzm_default.txt which is bundled.
    
    mzml_file, csv_file = simmzml(db, output_name, n=0, matrix=True, matrixmz=None)
    
    assert os.path.exists(mzml_file)
    assert os.path.exists(csv_file)
    
    # Check CSV content
    df = pd.read_csv(csv_file)
    assert len(df) == 0
    assert list(df.columns) == ['mz', 'rt', 'ins', 'sim_ins', 'name']
    
    # Check mzML content size (rough check)
    # The matrix simulation should generate significant data
    assert os.path.getsize(mzml_file) > 1000


def test_noise_peaks_injection(tmp_path):
    """Sharp chemical-noise injection should add peaks above the flat baseline
    in matrix channels (requires matrix=True) without erroring."""
    db = load_data('monahrms1')
    base = str(tmp_path / "no_noise")
    noisy = str(tmp_path / "with_noise")

    # Same seed, matrix on; only difference is injected noise peaks
    mz0, csv0 = simmzml(db, base, n=5, matrix=True, baseline=100,
                        baselinesd=30, seed=1, noise_peaks=0)
    mz1, csv1 = simmzml(db, noisy, n=5, matrix=True, baseline=100,
                        baselinesd=30, seed=1, noise_peaks=200,
                        noise_peak_sigma=(3, 15), noise_peak_snr=(5, 40))

    assert os.path.exists(mz1) and os.path.exists(csv1)
    # Same seed + same compounds => the only difference is the injected noise,
    # which adds positive signal, so total ion intensity must strictly increase.
    # The compound-peak ground truth (CSV) is unaffected.
    import pyteomics.mzml as pmz

    def total_intensity(path):
        s = 0.0
        with pmz.read(path) as reader:
            for spec in reader:
                ints = spec.get('intensity array')
                if ints is not None and len(ints):
                    s += float(ints.sum())
        return s

    assert total_intensity(mz1) > total_intensity(mz0)
    assert len(pd.read_csv(csv1)) == len(pd.read_csv(csv0))  # GT unchanged
