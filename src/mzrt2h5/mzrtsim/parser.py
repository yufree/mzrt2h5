import re
import numpy as np

def parse_msp(file_path):
    """
    Parses an MSP file and yields spectra dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        spectrum = {}
        peaks_mz = []
        peaks_intensity = []
        reading_peaks = False
        
        for line in f:
            line = line.strip()
            if not line:
                if spectrum:
                    if peaks_mz:
                        spectrum['mz'] = np.array(peaks_mz, dtype=float)
                        spectrum['intensity'] = np.array(peaks_intensity, dtype=float)
                        yield spectrum
                    spectrum = {}
                    peaks_mz = []
                    peaks_intensity = []
                    reading_peaks = False
                continue
            
            if ':' in line and not reading_peaks:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'Num Peaks':
                    reading_peaks = True
                    spectrum['num_peaks'] = int(value)
                else:
                    spectrum[key] = value
            elif reading_peaks:
                # Expecting "mz intensity" pair
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        peaks_mz.append(mz)
                        peaks_intensity.append(intensity)
                    except ValueError:
                        pass # Ignore malformed lines

        # Yield the last spectrum if exists
        if spectrum:
            if peaks_mz:
                spectrum['mz'] = np.array(peaks_mz, dtype=float)
                spectrum['intensity'] = np.array(peaks_intensity, dtype=float)
                yield spectrum

def filter_ms1(spectra_generator):
    """
    Filters a generator of spectra for MS1 only.
    """
    for spectrum in spectra_generator:
        if spectrum.get('Spectrum_type') == 'MS1':
            yield spectrum

def load_db(file_path, limit=None):
    """
    Loads database from MSP file, filtering for MS1.
    """
    db = []
    gen = filter_ms1(parse_msp(file_path))
    for i, s in enumerate(gen):
        if limit and i >= limit:
            break
        # Create a simplified object similar to the R list structure
        # name, idms, ionmode, prec, formula, np, rti, instr, msm, spectra(mz, ins)
        
        # Extract fields or use defaults
        # The R code uses 'ins' for notes/comments.
        
        entry = {
            'name': s.get('Name', ''),
            'idms': s.get('DB#', ''),
            'ionmode': s.get('Ion_mode', ''),
            'prec': float(s.get('ExactMass', 0)) if s.get('ExactMass') else 0.0,
            'formula': s.get('Formula', ''),
            'np': s.get('num_peaks', 0),
            'rti': 0, # Placeholder
            'instr': s.get('Instrument_type', ''),
            'msm': s.get('Comments', ''),
            'spectra': {
                'mz': s['mz'],
                'ins': s['intensity']
            }
        }
        db.append(entry)
    return db
