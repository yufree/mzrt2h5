import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from .writer import write_mzml
from .reader import SimpleMzMLReader
import os

def simmzml_background(blank_mzml, db, name, n=100, inscutoff=0.05, mzrange=(30, 1500),
                       ppm=5, sampleppm=5, mzdigit=5, noisesd=0.5, pwidth=10,
                       SNR=100, tailingfactor=1.2, compound=None, rtime=None,
                       tailingindex=None, seed=42, unique=False):
    """
    Simulates peaks and adds them to an existing blank mzML file.
    """
    np.random.seed(seed)
    
    # 1. Read RTs from blank
    print(f"Reading retention times from {blank_mzml}...")
    reader = SimpleMzMLReader(blank_mzml)
    rtime0 = reader.get_rts()
    if len(rtime0) == 0:
        raise ValueError("No retention times found in blank file.")
        
    print(f"Found {len(rtime0)} scans.")
    
    # 2. Prepare Compounds (Logic reused from simmzml)
    # Unique check
    if unique:
        seen = set()
        new_db = []
        for x in db:
            if x['name'] not in seen:
                seen.add(x['name'])
                new_db.append(x)
        db = new_db

    # Select compounds
    if compound is None:
        if n > len(db):
             n = len(db)
        indices = np.random.choice(len(db), n, replace=False)
        sub = [db[i] for i in indices]
    else:
        sub = [db[i] for i in compound]
        n = len(compound)
        
    # Retention times for compounds (Simulated)
    # Ensure they are within the range of the blank file
    min_rt, max_rt = np.min(rtime0), np.max(rtime0)
    
    if rtime is None:
        rtime = np.random.choice(rtime0, n, replace=True)
    elif len(rtime) != n:
        raise ValueError('Element numbers of retention time vector should have the same number of compounds.')

    # Peak widths
    if isinstance(pwidth, (int, float)):
        peakrange = poisson.rvs(pwidth, size=n)
    else:
        peakrange = np.array(pwidth)
        
    # SNR
    if isinstance(SNR, (int, float)):
        SNR_arr = np.full(n, SNR)
    else:
        if len(SNR) != n:
            SNR_arr = np.random.choice(SNR, n, replace=True)
        else:
            SNR_arr = np.array(SNR)

    # Chromatography simulation
    re_list = []
    for i in range(n):
        sigma = peakrange[i] / 4.0
        if sigma == 0: sigma = 0.1
        gaussian_peak = norm.pdf(rtime0, loc=rtime[i], scale=sigma)
        
        if np.max(gaussian_peak) > 0:
            gaussian_peak = gaussian_peak / np.max(gaussian_peak) * 100 * SNR_arr[i]
        
        sigma_tail = (2 * tailingfactor - 1) * peakrange[i] / 4.0
        if sigma_tail == 0: sigma_tail = 0.1
        tailing_peak = norm.pdf(rtime0, loc=rtime[i], scale=sigma_tail)
        
        if np.max(tailing_peak) > 0:
            tailing_peak = tailing_peak / np.max(tailing_peak) * 100 * SNR_arr[i]
        
        peak_idx = np.argmax(gaussian_peak)
        
        is_tailing = True
        if tailingindex is not None:
             if i not in tailingindex:
                 is_tailing = False
        
        final_peak = gaussian_peak.copy()
        if is_tailing:
            if peak_idx + 1 < len(rtime0):
                final_peak[peak_idx+1:] = tailing_peak[peak_idx+1:]
        
        re_list.append(final_peak)
    
    re = np.array(re_list)

    # Process spectra
    mz_list = []
    int_list = []
    
    for s in sub:
        m = s['spectra']['mz']
        i_ = s['spectra']['ins']
        
        if inscutoff is not None:
            max_i = np.max(i_) if len(i_) > 0 else 1
            idx = (i_ / max_i) > inscutoff
            m = m[idx]
            i_ = i_[idx]
        
        mz_list.append(m)
        int_list.append(i_)

    # 3. Combine and Merge with Blank
    subname = [s['name'] for s in sub]
    
    # Calculate simulated intensity profiles for all compounds
    # rem_list: list of (n_peaks_in_compound, n_scans)
    mzc_all = []
    rem_list = []
    
    for i in range(n):
        mzs = mz_list[i]
        ints = int_list[i]
        chrom = re[i]
        
        if len(mzs) == 0:
            continue
        
        profile = ints[:, np.newaxis] * chrom[np.newaxis, :]
        mzc_all.extend(np.round(mzs, mzdigit))
        rem_list.append(profile)

    rem = np.vstack(rem_list) if rem_list else np.array([])
    mzc = np.array(mzc_all)
    
    if len(rem) > 0:
        df_rem = pd.DataFrame(rem)
        df_rem['mz'] = mzc
        alld = df_rem.groupby('mz').sum()
        mzpeak = alld.index.values
        ins_peak_all = alld.values # (n_unique_sim_peaks, n_scans)
        
        mzpeak_noisy = mzpeak + np.random.normal(0, noisesd, len(mzpeak)) * mzpeak * 1e-6 * ppm
    else:
        mzpeak_noisy = np.array([])
        ins_peak_all = np.array([])

    # Iterate blank and merge
    print("Merging simulated peaks into blank data...")
    spectra_export = []
    reader = SimpleMzMLReader(blank_mzml)
    
    scan_idx = 0
    
    for spectrum in reader.get_spectra():
        if spectrum['ms_level'] != 1:
            # Just copy non-MS1
            spectra_export.append(spectrum)
            continue
            
        current_rt = spectrum['rt']
        
        # Get blank data
        mz_blank = spectrum['mz']
        int_blank = spectrum['intensity']
        
        # Get simulated data for this scan
        if len(mzpeak_noisy) > 0 and scan_idx < ins_peak_all.shape[1]:
            # Get intensities > 0
            sim_ints = ins_peak_all[:, scan_idx]
            mask = sim_ints > 0
            
            mz_sim = mzpeak_noisy[mask]
            int_sim = sim_ints[mask]
            
            # Apply sample ppm shift to sim peaks
            if len(mz_sim) > 0:
                 mz_sim = mz_sim + np.random.normal(0, noisesd, len(mz_sim)) * mz_sim * 1e-6 * sampleppm
            
            # Merge
            final_mz = np.concatenate([mz_blank, mz_sim])
            final_int = np.concatenate([int_blank, int_sim])
            
            # Sort
            sort_idx = np.argsort(final_mz)
            final_mz = final_mz[sort_idx]
            final_int = final_int[sort_idx]
            
            # Optional: Add noise to simulated peaks? 
            # (The logic in simmzml added noise to the *sum* of signal. 
            # Here blank already has noise. We assume simulated peaks are just added signal.)
            
            spectrum['mz'] = final_mz
            spectrum['intensity'] = final_int
        
        spectra_export.append(spectrum)
        scan_idx += 1
        
    write_mzml(name + '.mzML', spectra_export)
    
    # Generate simulated intensity profiles first to calculate max simulated intensity
    mzc_all = []
    rem_list = []
    sim_intensities = []  # Store simulated intensity max for each peak
    
    for i in range(n):
        mzs = mz_list[i]
        ints = int_list[i]
        chrom = re[i] # (n_scans,)
        
        if len(mzs) == 0:
            sim_intensities.append([])
            continue
            
        # Outer product: ints (n_peaks,) * chrom (n_scans,) -> (n_peaks, n_scans)
        profile = ints[:, np.newaxis] * chrom[np.newaxis, :]
        
        # Calculate max simulated intensity for each peak
        max_sim_ins = np.max(profile, axis=1)
        sim_intensities.append(max_sim_ins)
    
    # Generate CSV (Simulated compounds info)
    csv_mz = []
    csv_rt = []
    csv_ins = []
    csv_sim_ins = [] # Simulated intensity (max)
    csv_name = []
    
    for k in range(n):
        count = len(mz_list[k])
        csv_mz.extend(mz_list[k])
        csv_rt.extend([rtime[k]] * count)
        csv_ins.extend(int_list[k])
        # Add simulated intensity max for each peak
        if len(sim_intensities[k]) > 0:
            csv_sim_ins.extend(sim_intensities[k])
        else:
            csv_sim_ins.extend([0.0] * count)
        csv_name.extend([subname[k]] * count)

    df2 = pd.DataFrame({'mz': csv_mz, 'rt': csv_rt, 'ins': csv_ins, 'sim_ins': csv_sim_ins, 'name': csv_name})
    df2.to_csv(name + '.csv', index=False)
    
    return name + '.mzML', name + '.csv'

def simmzml(db, name, n=100, inscutoff=0.05, mzrange=(30, 1500), rtrange=(0, 600),
            ppm=5, sampleppm=5, mzdigit=5, noisesd=0.5, scanrate=0.2, pwidth=10,
            baseline=100, baselinesd=30, SNR=100, tailingfactor=1.2,
            compound=None, rtime=None, tailingindex=None, seed=42, unique=False,
            matrix=False, matrixmz=None):
    
    np.random.seed(seed)
    
    # Unique check
    if unique:
        seen = set()
        new_db = []
        for x in db:
            if x['name'] not in seen:
                seen.add(x['name'])
                new_db.append(x)
        db = new_db

    # Select compounds
    if compound is None:
        if n > len(db):
             n = len(db)
        # Random sample without replacement if n <= len(db)
        # R code uses sample(length(db), n). If replace=T is not specified, it's false.
        # But wait, does R sample with replacement by default? No.
        indices = np.random.choice(len(db), n, replace=False)
        sub = [db[i] for i in indices]
    else:
        # compound is list of indices
        sub = [db[i] for i in compound]
        n = len(compound)

    # Time points
    rtime0 = np.arange(rtrange[0], rtrange[1] + scanrate, scanrate)
    
    # Retention times for compounds
    if rtime is None:
        rtime = np.random.choice(rtime0, n, replace=True)
    elif len(rtime) != n:
        raise ValueError('Element numbers of retention time vector should have the same number of compounds.')
    
    # Peak widths
    if isinstance(pwidth, (int, float)):
        peakrange = poisson.rvs(pwidth, size=n)
    else:
        peakrange = np.array(pwidth)
        
    # SNR
    if isinstance(SNR, (int, float)):
        SNR_arr = np.full(n, SNR)
    else:
        if len(SNR) != n:
            SNR_arr = np.random.choice(SNR, n, replace=True)
        else:
            SNR_arr = np.array(SNR)

    # Chromatography simulation
    re_list = []
    for i in range(n):
        # Gaussian
        # stats::dnorm(rtime0, mean = rtime[i], sd = peakrange[i] / 4)
        sigma = peakrange[i] / 4.0
        if sigma == 0: sigma = 0.1 # avoid div by zero
        gaussian_peak = norm.pdf(rtime0, loc=rtime[i], scale=sigma)
        
        # Normalize and scale
        if np.max(gaussian_peak) > 0:
            gaussian_peak = gaussian_peak / np.max(gaussian_peak) * 100 * SNR_arr[i]
        
        # Tailing
        sigma_tail = (2 * tailingfactor - 1) * peakrange[i] / 4.0
        if sigma_tail == 0: sigma_tail = 0.1
        tailing_peak = norm.pdf(rtime0, loc=rtime[i], scale=sigma_tail)
        
        if np.max(tailing_peak) > 0:
            tailing_peak = tailing_peak / np.max(tailing_peak) * 100 * SNR_arr[i]
        
        peak_idx = np.argmax(gaussian_peak)
        
        # Merge
        # if tailingindex logic...
        # Here we simplify: if tailingindex is None, all tailing.
        is_tailing = True
        if tailingindex is not None:
             if i not in tailingindex: # Wait, i is 0 to n-1. tailingindex in R is 1-based usually? Assuming 0-based here for python
                 is_tailing = False
        
        final_peak = gaussian_peak.copy()
        if is_tailing:
            # Join: gaussian up to max, tailing after max
            # R: c(gaussian_peak[1:which.max], tailing_peak[which.max+1:end])
            if peak_idx + 1 < len(rtime0):
                final_peak[peak_idx+1:] = tailing_peak[peak_idx+1:]
        
        re_list.append(final_peak)
    
    re = np.array(re_list) # shape (n_compounds, n_scans)
    
    # Process spectra
    mz_list = []
    int_list = []
    
    for s in sub:
        m = s['spectra']['mz']
        i_ = s['spectra']['ins']
        
        if inscutoff is not None:
            max_i = np.max(i_) if len(i_) > 0 else 1
            idx = (i_ / max_i) > inscutoff
            m = m[idx]
            i_ = i_[idx]
        
        mz_list.append(m)
        int_list.append(i_)
        
    subname = [s['name'] for s in sub]
    
    # Combine chromatography and spectra first to calculate simulated intensity
    # rem: rows = all peaks from all compounds, cols = scans
    
    mzc_all = []
    rem_list = []
    sim_intensities = []  # Store simulated intensity max for each peak
    
    for i in range(n):
        mzs = mz_list[i]
        ints = int_list[i]
        chrom = re[i] # (n_scans,)
        
        if len(mzs) == 0:
            sim_intensities.append([])
            continue
            
        # Outer product: ints (n_peaks,) * chrom (n_scans,) -> (n_peaks, n_scans)
        # Using broadcasting
        profile = ints[:, np.newaxis] * chrom[np.newaxis, :]
        
        # Calculate max simulated intensity for each peak
        max_sim_ins = np.max(profile, axis=1)
        sim_intensities.append(max_sim_ins)
        
        mzc_all.extend(np.round(mzs, mzdigit))
        rem_list.append(profile)
    
    # Flatten for CSV
    # mzv, rtimev, namev, sim_insv
    csv_mz = []
    csv_rt = []
    csv_ins = [] # Database intensity
    csv_sim_ins = [] # Simulated intensity (max)
    csv_name = []
    
    for k in range(n):
        count = len(mz_list[k])
        csv_mz.extend(mz_list[k])
        csv_rt.extend([rtime[k]] * count)
        csv_ins.extend(int_list[k])
        # Add simulated intensity max for each peak
        if len(sim_intensities[k]) > 0:
            csv_sim_ins.extend(sim_intensities[k])
        else:
            csv_sim_ins.extend([0.0] * count)
        csv_name.extend([subname[k]] * count)

    df2 = pd.DataFrame({'mz': csv_mz, 'rt': csv_rt, 'ins': csv_ins, 'sim_ins': csv_sim_ins, 'name': csv_name})
        
    if not rem_list:
        if n > 0:
            raise ValueError("No peaks generated")
        else:
            # Handle 0 compounds case
            n_scans = len(rtime0)
            mzpeak = np.array([])
            mzpeak_noisy = np.array([])
            ins_peak = np.zeros((0, n_scans))
            final_mz = mzpeak_noisy
            allins = ins_peak
    else:
        rem = np.vstack(rem_list) # (total_peaks, n_scans)
        mzc = np.array(mzc_all)
        
        # Aggregate by m/z
        # Pandas is good for this
        df_rem = pd.DataFrame(rem)
        df_rem['mz'] = mzc
        alld = df_rem.groupby('mz').sum()
        
        mzpeak = alld.index.values
        
        # Add ppm noise to mzpeak base values (simulation of accurate mass measurement error?)
        # R: mzpeak + rnorm(...) * mzpeak * 1e-6 * ppm
        mzpeak_noisy = mzpeak + np.random.normal(0, noisesd, len(mzpeak)) * mzpeak * 1e-6 * ppm
        
        n_unique_peaks = len(mzpeak)
        n_scans = len(rtime0)
        
        # Calculate noise for peaks
        noise_peak = np.random.normal(baseline, baselinesd, (n_unique_peaks, n_scans))
        ins_peak = alld.values + noise_peak
        
        final_mz = mzpeak_noisy
        allins = ins_peak
    
    if matrix:
        if matrixmz is None:
            # Load default mzm
            try:
                module_dir = os.path.dirname(__file__)
                mzm_path = os.path.join(module_dir, 'mzm_default.txt')
                mzm = np.loadtxt(mzm_path)
            except Exception:
                print("Warning: Could not load default matrix m/z data. Skipping matrix simulation.")
                mzm = np.array([])
        else:
            mzm = np.array(matrixmz)
            
        if len(mzm) > 0:
            mzm = np.round(mzm, mzdigit)
            # Remove matrix peaks that overlap with compound peaks (exact match after rounding)
            # R: mzmatrix <- mzm[!mzm%in%mzpeak]
            # using set difference
            mzpeak_rounded = np.round(mzpeak, mzdigit)
            mask_unique = ~np.isin(mzm, mzpeak_rounded)
            mzmatrix = mzm[mask_unique]
            
            if len(mzmatrix) > 0:
                # Generate matrix intensity
                # insmatrix <- matrix(rnorm(..., mean=baseline, sd=baselinesd)...)
                ins_matrix = np.random.normal(baseline, baselinesd, (len(mzmatrix), n_scans))
                
                # Combine
                # R: allins <- rbind(as.matrix(inspeak), insmatrix)
                #    mz <- c(mzpeak, mzmatrix)
                
                # We need to combine mz arrays and intensity arrays
                # final_mz currently holds the "noisy" peak m/z. 
                # The matrix m/z should technically also have noise? 
                # R code:
                # mz <- c(mzpeak, mzmatrix)
                # idx <- order(mz)
                # mz <- mz[idx]
                # allins <- allins[idx,]
                # Then later in the loop: mzt + rnorm(...) * mzt ...
                
                # So we combine now.
                combined_mz = np.concatenate([final_mz, mzmatrix])
                combined_ins = np.vstack([allins, ins_matrix])
                
                # Sort
                sort_idx = np.argsort(combined_mz)
                final_mz = combined_mz[sort_idx]
                allins = combined_ins[sort_idx, :]

    
    spectra_export = []
    
    for i in range(n_scans):
        ins_scan = allins[:, i]
        
        # Filter
        mask = (ins_scan > 0) & (final_mz > mzrange[0]) & (final_mz < mzrange[1])
        
        mzt = final_mz[mask]
        inst = ins_scan[mask]
        
        # Add sample ppm shift (jitter per scan)
        # mzt + rnorm(...) * mzt * 1e-6 * sampleppm
        if len(mzt) > 0:
            mzt = mzt + np.random.normal(0, noisesd, len(mzt)) * mzt * 1e-6 * sampleppm
            
        # Sort by m/z
        sort_idx = np.argsort(mzt)
        mzt = mzt[sort_idx]
        inst = inst[sort_idx]
        
        spectra_export.append({
            'mz': mzt,
            'intensity': inst,
            'rt': rtime0[i],
            'id': f"scan={i+1}",
            'ms_level': 1
        })

    # Write mzML
    write_mzml(name + '.mzML', spectra_export)
    
    # Write CSV
    df2.to_csv(name + '.csv', index=False)
    
    return name + '.mzML', name + '.csv'

