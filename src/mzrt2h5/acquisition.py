"""Download public metabolomics datasets (MetaboLights) into a local folder.

Folds the standalone data/acquisition/ shell scripts into the package as a
first-class feature, so the full path is one tool:

    mzrt2h5 download MTBLS266 ./MTBLS266 --pol pos      # flat mzML study
    mzrt2h5 download MTBLS364 ./MTBLS364 --ext .raw.zip --unzip   # Waters vendor raw
    mzrt2h5 process  ./MTBLS266 out.h5 --metadata-csv-path ./MTBLS266

Design:
- Lists spectra by **recursively walking the EBI FTP mirror** (Apache autoindex).
  This is more robust than the MetaboLights REST API, whose assayTable comes back
  empty for some studies (e.g. MTBLS364) and which doesn't expose vendor-raw paths.
- Pulls ISA-Tab metadata (s_/a_/i_/m_) at the study root, so `process
  --metadata-format isatab` works straight after.
- Resumable: a file already present at the expected size (or an already-unzipped
  vendor dir) is skipped on re-run.
- stdlib only (urllib + zipfile); no new dependency.
"""
import os
import re
import sys
import shutil
import socket
import zipfile
import subprocess
import urllib.request
from html.parser import HTMLParser

# urlretrieve() has no per-call timeout; without this a silently dropped
# server connection hangs the whole download forever (seen on EBI FTP).
# 300 s of no progress is plenty for any single chunk.
socket.setdefaulttimeout(300)

FTP_ROOT = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"

# Metabolomics Workbench
MW_BASE = "https://www.metabolomicsworkbench.org"
MW_REST = MW_BASE + "/rest"
# named sections in the per-study .7z bundles (filename fragment after "ST<id>_")
MW_SECTIONS = {
    "mzml_centroid": "mzML_centroid",
    "mzml_profile": "mzML_profile",
    "raw": "data_raw",
    "metadata": "sample_metadata",
}

# default spectra extensions; vendor-raw studies pass their own via --ext
DEFAULT_EXTS = (".mzml",)


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.hrefs.append(v)


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "mzrt2h5/acquisition"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _list_dir(url):
    """Return (files, subdirs) hrefs from an Apache autoindex page at url/."""
    try:
        html = _get(url).decode("utf-8", "replace")
    except Exception:
        return [], []
    p = _LinkParser()
    p.feed(html)
    files, dirs = [], []
    for h in p.hrefs:
        if h.startswith("?") or h.startswith("/") or h.startswith("http"):
            continue  # sort links, parent dir, absolute
        if h == "../":
            continue
        if h.endswith("/"):
            dirs.append(h.rstrip("/"))
        else:
            files.append(h)
    return files, dirs


def _walk_spectra(base_url, exts, polarity, _depth=0, _max_depth=6):
    """Yield relative paths (from base_url) of files matching exts + polarity."""
    files, dirs = _list_dir(base_url + "/")
    for f in files:
        low = f.lower()
        if not any(low.endswith(e) for e in exts):
            continue
        if polarity == "pos" and "pos" not in low:
            continue
        if polarity == "neg" and "neg" not in low:
            continue
        yield f
    if _depth < _max_depth:
        for d in dirs:
            for rel in _walk_spectra(f"{base_url}/{d}", exts, polarity,
                                     _depth + 1, _max_depth):
                yield f"{d}/{rel}"


def list_study_spectra(study_id, polarity="all", exts=DEFAULT_EXTS, subpath="FILES"):
    """List spectra file paths (relative to study root) for a MetaboLights study.

    Args:
        study_id: e.g. "MTBLS266".
        polarity: "all" | "pos" | "neg" (filter by filename substring).
        exts: tuple of lowercase extensions to collect (e.g. (".mzml",) or
              (".raw.zip",)).
        subpath: where under the study root to start walking (default "FILES").

    Returns:
        list[str] relative paths like "FILES/RAW_FILES/HILIC/POS/x.raw.zip".
    """
    exts = tuple(e.lower() for e in exts)
    base = f"{FTP_ROOT}/{study_id}/{subpath}"
    return [f"{subpath}/{rel}" for rel in _walk_spectra(base, exts, polarity)]


def _fetch_isatab(study_id, dest):
    """Download ISA-Tab metadata (s_/a_/i_/m_*.txt|tsv) to dest. Returns count."""
    files, _ = _list_dir(f"{FTP_ROOT}/{study_id}/")
    n = 0
    for f in files:
        if re.match(r"^(s_|a_|i_|m_).*\.(txt|tsv)$", f):
            try:
                data = _get(f"{FTP_ROOT}/{study_id}/{f}")
                with open(os.path.join(dest, f), "wb") as fh:
                    fh.write(data)
                n += 1
            except Exception as e:
                print(f"  [meta] failed {f}: {e}", file=sys.stderr)
    return n


def _remote_size(url):
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "mzrt2h5/acquisition"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def download_study(study_id, dest, polarity="all", exts=DEFAULT_EXTS,
                   subpath="FILES", unzip=False, progress=True):
    """Download a MetaboLights study's spectra + ISA-Tab metadata into dest.

    Mirrors the remote directory layout under dest (so nested vendor-raw studies
    like MTBLS364 keep their HILIC/RPLC × POS/NEG structure). Resumable.

    Args:
        study_id: e.g. "MTBLS364".
        dest: local output directory (created if missing).
        polarity: "all" | "pos" | "neg".
        exts: extensions to fetch ((".mzml",) default; (".raw.zip",) for Waters).
        subpath: study subpath to walk (default "FILES").
        unzip: if True, unzip *.zip archives in place and delete the archive.
        progress: print per-file progress.

    Returns:
        dict: {'study_id', 'n_total', 'n_present', 'dest', 'isatab_files'}.
    """
    os.makedirs(dest, exist_ok=True)
    n_meta = _fetch_isatab(study_id, dest)
    if progress:
        print(f"[meta] {study_id}: {n_meta} ISA-Tab files -> {dest}")

    rels = list_study_spectra(study_id, polarity=polarity, exts=exts, subpath=subpath)
    n = len(rels)
    if progress:
        print(f"[list] {study_id}: {n} spectra files (polarity={polarity}, "
              f"exts={list(exts)})")
    if n == 0:
        return {"study_id": study_id, "n_total": 0, "n_present": 0,
                "dest": dest, "isatab_files": n_meta}

    present = 0
    for i, rel in enumerate(rels, 1):
        url = f"{FTP_ROOT}/{study_id}/{rel}"
        out = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)

        # already unzipped vendor dir? (e.g. x.raw.zip -> x.raw/)
        if unzip and out.endswith(".zip"):
            unzipped = out[:-4]  # strip .zip -> .../x.raw
            if os.path.isdir(unzipped):
                present += 1
                if progress:
                    print(f"  [{i}/{n}] {rel} (unzipped, skip)")
                continue

        # already complete?
        if os.path.exists(out):
            rsize = _remote_size(url)
            if rsize is None or os.path.getsize(out) == rsize:
                if not (unzip and out.endswith(".zip")):
                    present += 1
                    if progress:
                        print(f"  [{i}/{n}] {rel} (present, skip)")
                    continue

        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f"  [{i}/{n}] FAIL {rel}: {e}", file=sys.stderr)
            continue

        if unzip and out.endswith(".zip"):
            try:
                with zipfile.ZipFile(out) as z:
                    z.extractall(os.path.dirname(out))
                os.remove(out)
            except Exception as e:
                print(f"  [{i}/{n}] BADZIP {rel}: {e}", file=sys.stderr)
                continue
        present += 1
        if progress:
            print(f"  [{i}/{n}] {rel}")

    if progress:
        print(f"DONE {study_id}: {present}/{n} present -> {dest}")
    return {"study_id": study_id, "n_total": n, "n_present": present,
            "dest": dest, "isatab_files": n_meta}


# --------------------------------------------------------------------------- #
# Metabolomics Workbench
#
# Different model from MetaboLights: no per-file FTP index. Each study offers a
# handful of bundled .7z archives (doc / sample_metadata / data_mzML_centroid /
# data_mzML_profile / data_raw / resources), listed on the SetupRawDataDownload
# page. mwTab metadata comes from the REST API. Many studies ship ready-made
# mzML (centroid), so no msconvert step is needed.
# --------------------------------------------------------------------------- #

def fetch_workbench_mwtab(study_id, dest):
    """Download the mwTab metadata for a Workbench study to dest/<ID>.mwtab.txt.

    Returns the written path, or None on failure. Use with
    `process --metadata-format mwtab`.
    """
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, f"{study_id}.mwtab.txt")
    try:
        data = _get(f"{MW_REST}/study/study_id/{study_id}/mwtab/txt")
        with open(out, "wb") as fh:
            fh.write(data)
        return out
    except Exception as e:
        print(f"  [meta] mwTab fetch failed: {e}", file=sys.stderr)
        return None


def list_workbench_bundles(study_id):
    """List downloadable bundle files for a Workbench study.

    Returns list[(filename, url)] parsed from the SetupRawDataDownload page
    (the /studydownload/<ID>_*.7z archives plus any Results.txt).
    """
    url = (f"{MW_BASE}/data/DRCCStudySummary.php"
           f"?Mode=SetupRawDataDownload&StudyID={study_id}")
    html = _get(url).decode("utf-8", "replace")
    p = _LinkParser()
    p.feed(html)
    out = []
    seen = set()
    for h in p.hrefs:
        if "/studydownload/" not in h:
            continue
        fn = h.rsplit("/", 1)[-1]
        if fn in seen:
            continue
        seen.add(fn)
        full = h if h.startswith("http") else MW_BASE + h
        out.append((fn, full))
    return out


def _extract_7z(path, outdir):
    """Extract a .7z archive via py7zr or a 7z CLI. Returns True on success."""
    try:
        import py7zr
        with py7zr.SevenZipFile(path, "r") as z:
            z.extractall(outdir)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  [extract] py7zr failed: {e}", file=sys.stderr)
    for cli in ("7z", "7za", "7zz"):
        if shutil.which(cli):
            try:
                subprocess.run([cli, "x", "-y", f"-o{outdir}", path], check=True)
                return True
            except Exception as e:
                print(f"  [extract] {cli} failed: {e}", file=sys.stderr)
    return False


def download_workbench_study(study_id, dest, section="mzml_centroid",
                             extract=True, progress=True):
    """Download a Metabolomics Workbench study bundle into dest.

    Args:
        study_id: e.g. "ST004504".
        dest: local output directory.
        section: which bundle to fetch — one of
            "mzml_centroid" (ready mzML, default), "mzml_profile",
            "raw" (vendor raw), "metadata" (tiny), or "all".
            The small sample_metadata bundle + mwTab are always fetched.
        extract: extract .7z after download (needs py7zr or a 7z CLI; if neither,
            the .7z is kept and a message is printed).
        progress: print progress.

    Returns:
        dict: {'study_id', 'dest', 'mwtab', 'downloaded': [filenames], 'extracted': bool}.
    """
    os.makedirs(dest, exist_ok=True)
    if section not in MW_SECTIONS and section != "all":
        raise ValueError(f"section must be 'all' or one of {list(MW_SECTIONS)}")

    mwtab = fetch_workbench_mwtab(study_id, dest)
    if progress:
        print(f"[meta] {study_id}: mwTab -> {mwtab}")

    bundles = list_workbench_bundles(study_id)
    if progress:
        print(f"[list] {study_id}: {len(bundles)} bundles available: "
              f"{[b[0] for b in bundles]}")

    # always include the tiny sample_metadata bundle; add the requested section
    wanted_fragments = {MW_SECTIONS["metadata"]}
    if section == "all":
        wanted_fragments |= set(MW_SECTIONS.values())
    else:
        wanted_fragments.add(MW_SECTIONS[section])

    pick = [(fn, u) for (fn, u) in bundles
            if any(frag in fn for frag in wanted_fragments)]
    if not pick:
        print(f"  [warn] no bundle matched section='{section}'. Available: "
              f"{[b[0] for b in bundles]}", file=sys.stderr)

    downloaded, any_extracted = [], False
    for fn, url in pick:
        out = os.path.join(dest, fn)
        rsize = _remote_size(url)
        if os.path.exists(out) and rsize and os.path.getsize(out) == rsize:
            if progress:
                print(f"  {fn} (present, skip)")
        else:
            if progress:
                gb = f"{rsize/1e9:.1f} GB" if rsize else "size unknown"
                print(f"  downloading {fn} ({gb})...")
            try:
                urllib.request.urlretrieve(url, out)
            except Exception as e:
                print(f"  FAIL {fn}: {e}", file=sys.stderr)
                continue
        downloaded.append(fn)

        if extract and fn.endswith(".7z"):
            if _extract_7z(out, dest):
                any_extracted = True
                if progress:
                    print(f"  extracted {fn}")
            elif progress:
                print(f"  [extract] no py7zr / 7z CLI — kept {fn}; "
                      f"`pip install py7zr` or `brew install p7zip` then re-run.")

    if progress:
        print(f"DONE {study_id}: {len(downloaded)} bundle(s) -> {dest}")
    return {"study_id": study_id, "dest": dest, "mwtab": mwtab,
            "downloaded": downloaded, "extracted": any_extracted}


# --------------------------------------------------------------------------- #
# Turnkey: download -> build HDF5 (only when the download yields mzML)
# --------------------------------------------------------------------------- #

def _find_mzml(folder):
    return [os.path.join(r, f) for r, _, fs in os.walk(folder)
            for f in fs if f.lower().endswith(".mzml")]


def _metadata_for(study_id, dest):
    """(metadata_path, format) for a downloaded study, by accession prefix.

    MetaboLights -> the ISA-Tab dir (dest); Workbench -> the dest/<ID>.mwtab.txt.
    """
    sid = study_id.upper()
    if sid.startswith("MTBLS"):
        return dest, "isatab"
    if sid.startswith("ST"):
        return os.path.join(dest, f"{study_id}.mwtab.txt"), "mwtab"
    raise ValueError(f"Unrecognized accession '{study_id}' (expected MTBLS*/ST*).")


def build_h5_from_download(study_id, dest, h5_path,
                           rt_precision=0.1, mz_precision=0.0001,
                           min_rel_intensity=None, progress=True):
    """Build the sparse HDF5 store from an already-downloaded study folder.

    Requires mzML on disk under `dest` (i.e. the study shipped mzML, or you have
    already converted vendor raw with msconvert). Auto-selects the metadata
    file/format from the accession (ISA-Tab dir for MTBLS*, mwTab file for ST*).

    Raises RuntimeError if no mzML are found (vendor-raw needs conversion first).

    Returns:
        dict: {'h5_path', 'n_mzml', 'metadata_format'}.
    """
    from .processing import save_dataset_as_sparse_h5

    mzml = _find_mzml(dest)
    if not mzml:
        raise RuntimeError(
            f"No mzML found under {dest}. The study shipped vendor raw — convert "
            f"to mzML (msconvert) first, then build. (MTBLS364 / ST*_data_raw.)")

    meta_path, fmt = _metadata_for(study_id, dest)
    if not os.path.exists(meta_path):
        raise RuntimeError(f"Metadata not found at {meta_path}; re-run download.")

    if progress:
        print(f"[build] {len(mzml)} mzML, metadata={fmt} -> {h5_path}")
    save_dataset_as_sparse_h5(
        folder=dest, save_path=h5_path,
        rt_precision=rt_precision, mz_precision=mz_precision,
        metadata_csv_path=meta_path, format=fmt,
        min_rel_intensity=min_rel_intensity,
    )
    return {"h5_path": h5_path, "n_mzml": len(mzml), "metadata_format": fmt}
