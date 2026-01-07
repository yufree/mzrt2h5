import os
import sys
import numpy as np
import pandas as pd
import uuid
import tempfile

# Import from internal mzrtsim module
from .mzrtsim import load_data, simmzml
from .mzrtsim.sim import simmzml_background


def generate_simulation_data(
    n_compounds=100,
    inscutoff=0.05,
    mzrange=(100, 1000),
    rtrange=(0, 600),
    ppm=5,
    sampleppm=5,
    mzdigit=5,
    noise_sd=0.5,
    scanrate=0.1,
    pwidth=10,
    baseline=100,
    baselinesd=30,
    snr=100,
    tailing_factor=1.2,
    compound=None,
    rtime=None,
    tailingindex=None,
    seed=42,
    unique=False,
    matrix=False,
    matrixmz=None,
    output_dir="simulation_output"
):
    """
    Generate simulation data using mzrtsim package.
    
    Args:
        n_compounds (int): Number of compounds to simulate
        inscutoff (float): Intensity cutoff
        mzrange (tuple): m/z range (min, max)
        rtrange (tuple): Retention time range (min, max)
        ppm (float): Mass accuracy in ppm
        sampleppm (float): Sample mass accuracy in ppm
        mzdigit (int): Number of decimal digits for m/z values
        noise_sd (float): Noise standard deviation
        scanrate (float): Scan rate
        pwidth (float or list/array): Peak width. Can be a scalar or a vector of length n_compounds.
        baseline (float or list/array): Baseline intensity. Can be a scalar or a vector of length n_scans to simulate baseline shift.
        baselinesd (float): Baseline standard deviation
        snr (int, float, or list/array): Signal-to-noise ratio. Can be a scalar or a vector of length n_compounds.
        tailing_factor (float): Peak tailing factor
        compound (list): Specific compounds to use (None for random selection)
        rtime (list/array): Specific retention times (None for random). Must be a vector of length n_compounds if provided.
        tailingindex (list): Indices (0-based) of compounds with tailing (None for all).
        seed (int): Random seed
        unique (bool): Whether to use unique compounds
        matrix (bool): Whether to generate matrix
        matrixmz (list): Matrix m/z values
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Paths to generated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load database
    print("Loading database...")
    db = load_data('monahrms1')
    
    # Set all simulation parameters
    params = {
        'n': n_compounds,
        'inscutoff': inscutoff,
        'mzrange': mzrange,
        'rtrange': rtrange,
        'ppm': ppm,
        'sampleppm': sampleppm,
        'mzdigit': mzdigit,
        'noisesd': noise_sd,
        'scanrate': scanrate,
        'pwidth': pwidth,
        'baseline': baseline,
        'baselinesd': baselinesd,
        'SNR': snr,
        'tailingfactor': tailing_factor,
        'compound': compound,
        'rtime': rtime,
        'tailingindex': tailingindex,
        'seed': seed,
        'unique': unique,
        'matrix': matrix,
        'matrixmz': matrixmz
    }
    
    # Generate unique filename without data_type
    filename_prefix = f"simulation_{uuid.uuid4().hex[:8]}"
    
    # Run simulation
    print(f"Generating simulation data...")
    output_path = os.path.join(output_dir, filename_prefix)
    simmzml(db, output_path, **params)
    
    # Prepare output paths
    mzml_path = f"{output_path}.mzML"
    csv_path = f"{output_path}.csv"
    
    return {
        "mzml_path": mzml_path,
        "csv_path": csv_path,
        "params": params
    }


def simulate_background(
    blank_mzml_path,
    data_type="background",
    n_compounds=50,
    noise_sd=0.5,
    snr=100,
    tailing_factor=1.2,
    seed=42,
    output_dir="simulation_output"
):
    """
    Simulate peaks on top of a blank mzML file.
    
    Args:
        blank_mzml_path (str): Path to blank mzML file
        data_type (str): Type of simulation data to generate
        n_compounds (int): Number of compounds to simulate
        noise_sd (float): Noise standard deviation
        snr (int): Signal-to-noise ratio
        tailing_factor (float): Peak tailing factor
        seed (int): Random seed
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Paths to generated files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load database
    print("Loading database...")
    db = load_data('monahrms1')
    
    # Generate unique filename
    filename_prefix = f"{data_type}_{uuid.uuid4().hex[:8]}"
    output_path = os.path.join(output_dir, filename_prefix)
    
    # Run simulation with background
    print(f"Generating background simulation...")
    simmzml_background(
        blank_mzml_path, db, output_path,
        n=n_compounds,
        noisesd=noise_sd,
        SNR=snr,
        tailingfactor=tailing_factor,
        seed=seed
    )
    
    # Prepare output paths
    mzml_path = f"{output_path}.mzML"
    csv_path = f"{output_path}.csv"
    
    return {
        "mzml_path": mzml_path,
        "csv_path": csv_path,
        "data_type": data_type
    }


if __name__ == "__main__":
    # Example usage
    print("Generating complex high-resolution simulation data...")
    result = generate_simulation_data(
        data_type="complex_hires",
        n_compounds=100,
        output_dir="test_simulation"
    )
    
    print(f"Generated files:")
    print(f"  mzML: {result['mzml_path']}")
    print(f"  CSV: {result['csv_path']}")
