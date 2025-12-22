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

