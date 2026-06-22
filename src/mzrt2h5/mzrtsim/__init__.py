"""mzrtsim — LC-MS data simulation.

Python port of the mzrtsim R package. Cite the original method:

    mzrtsim, Anal. Chem. 2025, 97 (32), 17309-17314.
    DOI: 10.1021/acs.analchem.5c01213

The simulator generates mzML with known composition, retention times, peak
shapes, and noise — providing the ground truth that drives the suite's
quantifiable validation (real metabolomics data has no ground truth). `simmzml`
adds sharp chemical-noise channels (noise_peaks / noise_peak_sigma /
noise_peak_snr, requires matrix=True) so peak-quality scoring is tested against
realistic non-analyte peaks rather than only a flat Gaussian baseline.
"""
from .sim import simmzml
from .parser import parse_msp
import os
import pickle

def load_data(name='monahrms1'):
    """
    Loads a bundled database.
    
    Args:
        name (str): Name of the database to load. Currently only 'monahrms1' is available.
        
    Returns:
        list: The loaded database (list of spectra dictionaries).
    """
    if name == 'monahrms1':
        # Path relative to this file
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path, 'data', 'monahrms1.pkl')
        
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Database file not found at {data_path}. Please run create_db.py to generate it.")
    else:
        raise ValueError(f"Unknown database name: {name}")

