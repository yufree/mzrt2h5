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
