#!/usr/bin/env python3

"""
Test script to verify the addition of simulated intensity column in CSV output
"""

import os
import pandas as pd
import tempfile
from mzrt2h5.mzrtsim import load_data, simmzml

def test_simulated_intensity_column():
    """Test that CSV output contains simulated intensity column"""
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_name = os.path.join(tmpdir, "test_sim_ins")
        
        # Load database
        db = load_data('monahrms1')
        
        # Run simulation with small number of compounds
        print("Running simulation...")
        mzml_file, csv_file = simmzml(db, output_name, n=5, matrix=False)
        
        # Check if files were created
        assert os.path.exists(mzml_file), f"mzML file not created: {mzml_file}"
        assert os.path.exists(csv_file), f"CSV file not created: {csv_file}"
        
        # Read and check CSV content
        print(f"Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Check columns
        expected_columns = ['mz', 'rt', 'ins', 'sim_ins', 'name']
        assert list(df.columns) == expected_columns, \
            f"Expected columns {expected_columns}, got {list(df.columns)}"
        
        # Check that sim_ins column has reasonable values
        assert 'sim_ins' in df.columns
        assert df['sim_ins'].dtype in ['float64', 'float32', 'int64', 'int32']
        assert len(df) > 0, "CSV file is empty"
        
        # Print sample data
        print("\nSample CSV content:")
        print(df.head())
        
        # Print column descriptions
        print("\nColumn descriptions:")
        for col in df.columns:
            print(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")
        
        print("\n✅ Test passed! CSV file contains simulated intensity column 'sim_ins'.")

if __name__ == "__main__":
    test_simulated_intensity_column()
