import click
from .processing import save_dataset_as_sparse_h5, analyze_ms1_ms2_response
from .visualization import plot_sample_image, plot_ms1ms2_response
from .acquisition import download_study, download_workbench_study, build_h5_from_download

@click.group()
def main():
    """A command-line tool for processing mzML files and visualizing mass spec data."""
    pass


@main.command()
@click.argument('study_id', type=str)
@click.argument('dest', type=click.Path(file_okay=False, resolve_path=True))
@click.option('--pol', 'polarity', type=click.Choice(['all', 'pos', 'neg']),
              default='all', show_default=True,
              help='[MetaboLights] Polarity subset (filtered by filename substring).')
@click.option('--ext', 'exts', multiple=True,
              help='[MetaboLights] Spectra extension(s) to fetch. Default .mzML. '
                   'Use e.g. "--ext .raw.zip" for Waters vendor-raw studies (MTBLS364).')
@click.option('--subpath', default='FILES', show_default=True,
              help='[MetaboLights] Study subpath to walk (e.g. FILES/RAW_FILES/HILIC).')
@click.option('--unzip', is_flag=True, default=False,
              help='[MetaboLights] Unzip *.zip archives in place (for vendor .raw.zip).')
@click.option('--section', type=click.Choice(
                  ['mzml_centroid', 'mzml_profile', 'raw', 'metadata', 'all']),
              default='mzml_centroid', show_default=True,
              help='[Workbench] Which .7z bundle to fetch.')
@click.option('--no-extract', 'no_extract', is_flag=True, default=False,
              help='[Workbench] Keep the .7z archive instead of extracting it.')
@click.option('--h5', 'h5_path', type=click.Path(writable=True, resolve_path=True),
              default=None,
              help='Turnkey: after download, build this HDF5 store from the mzML '
                   '(auto-picks ISA-Tab/mwTab metadata). Skipped if only vendor raw.')
@click.option('--rt-precision', default=0.1, type=float, show_default=True,
              help='[--h5] RT bin size (seconds).')
@click.option('--mz-precision', default=0.0001, type=float, show_default=True,
              help='[--h5] m/z bin size (Da).')
@click.option('--min-rel-intensity', type=float, default=None,
              help='[--h5] Drop points below this fraction of each scan base peak '
                   '(profile-mode denoising, e.g. 0.001).')
def download(study_id, dest, polarity, exts, subpath, unzip, section, no_extract,
             h5_path, rt_precision, mz_precision, min_rel_intensity):
    """Download a public study (spectra + metadata) into DEST.

    The repository is auto-detected from the accession prefix:

    \b
      MTBLS* -> MetaboLights (recursive EBI FTP walk; --pol/--ext/--subpath/--unzip)
      ST*     -> Metabolomics Workbench (.7z bundles; --section/--no-extract)

    With --h5 it also builds the HDF5 store in one step (when the download yields
    mzML; vendor raw needs an msconvert step first). Otherwise build later with
    `mzrt2h5 process DEST out.h5 --metadata-csv-path ...`.

    Examples:

    \b
      mzrt2h5 download MTBLS266 ./MTBLS266 --pol pos --h5 mtbls266.h5
      mzrt2h5 download MTBLS364 ./MTBLS364 --ext .raw.zip --unzip
      mzrt2h5 download ST004504 ./ST004504 --section mzml_centroid --h5 st004504.h5
    """
    sid = study_id.upper()
    try:
        if sid.startswith('MTBLS'):
            exts = exts if exts else ('.mzML',)
            res = download_study(study_id, dest, polarity=polarity, exts=exts,
                                 subpath=subpath, unzip=unzip)
            click.echo(click.style(
                f"Downloaded {res['n_present']}/{res['n_total']} files "
                f"(+{res['isatab_files']} metadata) to {res['dest']}", fg='green'))
        elif sid.startswith('ST'):
            res = download_workbench_study(study_id, dest, section=section,
                                          extract=not no_extract)
            click.echo(click.style(
                f"Downloaded {len(res['downloaded'])} bundle(s) "
                f"(extracted={res['extracted']}) to {res['dest']}", fg='green'))
        else:
            raise click.BadParameter(
                f"Unrecognized accession '{study_id}'. Expected MTBLS* "
                f"(MetaboLights) or ST* (Metabolomics Workbench).")

        if h5_path:
            try:
                b = build_h5_from_download(
                    study_id, dest, h5_path,
                    rt_precision=rt_precision, mz_precision=mz_precision,
                    min_rel_intensity=min_rel_intensity)
                click.echo(click.style(
                    f"Built HDF5 from {b['n_mzml']} mzML ({b['metadata_format']}): "
                    f"{b['h5_path']}", fg='green'))
            except RuntimeError as e:
                click.echo(click.style(f"Download done; HDF5 build skipped: {e}",
                                       fg='yellow'))
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(click.style(f"Download failed: {e}", fg='red'), err=True)
        raise SystemExit(1)

@main.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument('save_path', type=click.Path(writable=True, resolve_path=True))
@click.option('--rt-precision', default=0.1, type=float, help='Bin size for the retention time axis (seconds).')
@click.option('--mz-precision', default=0.0001, type=float, help='Bin size for the m/z axis (Da). Default 0.0001 Da gives <0.5 ppm accuracy at m/z 500, enabling ppm-level HR restoration by mzrtpeak.')
@click.option('--metadata-csv-path', type=click.Path(exists=True, resolve_path=True), required=True, help='Path to metadata file (CSV, mwTab, or ISA-Tab directory).')
@click.option('--metadata-format', type=click.Choice(['auto', 'csv', 'mwtab', 'isatab'], case_sensitive=False), default='auto', help='Metadata file format. Default: auto-detect.')
@click.option('--sample-id-col', default='Sample Name', help='Column name for sample IDs (CSV/TSV only).')
@click.option('--separator', default=',', help='Separator for CSV/TSV metadata files.')
@click.option('--mz-range', type=(float, float), help='Fixed (min, max) m/z range.')
@click.option('--rt-range', type=(float, float), help='Fixed (min, max) RT range.')
@click.option('--min-rel-intensity', type=float, default=None,
              help='Keep only points >= this fraction of each scan base peak (e.g. 0.001). '
                   'Recommended for profile-mode data (e.g. QTOF) to denoise and bound HDF5 size; '
                   'leave unset for centroided data.')
def process(folder, save_path, rt_precision, mz_precision, metadata_csv_path, metadata_format, sample_id_col, separator, mz_range, rt_range, min_rel_intensity):
    """Processes a folder of mzML files and saves them as a single, consolidated sparse HDF5 file."""

    # Click returns empty tuples for non-provided tuple options, convert to None
    mz_range = mz_range if mz_range else None
    rt_range = rt_range if rt_range else None

    fmt = None if metadata_format == 'auto' else metadata_format

    click.echo(f"Starting processing of folder: {folder}")

    try:
        save_dataset_as_sparse_h5(
            folder=folder,
            save_path=save_path,
            rt_precision=rt_precision,
            mz_precision=mz_precision,
            metadata_csv_path=metadata_csv_path,
            sample_id_col=sample_id_col,
            separator=separator,
            format=fmt,
            mz_range=mz_range,
            rt_range=rt_range,
            min_rel_intensity=min_rel_intensity
        )
        click.echo(click.style(f"Successfully created HDF5 file at: {save_path}", fg='green'))
    except Exception as e:
        click.echo(click.style(f"An error occurred: {e}", fg='red'), err=True)

@main.command()
@click.argument('h5_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('sample_id', type=str)
@click.option('--rt-precision', default=0.5, type=float, help='RT precision for image reconstruction.')
@click.option('--mz-precision', default=0.05, type=float, help='m/z precision for image reconstruction.')
@click.option('--output-path', type=click.Path(writable=True, resolve_path=True), help='Path to save the plot (e.g., image.png). If not provided, displays interactively.')
@click.option('--cmap', default='viridis', help='Colormap for the plot (e.g., viridis, plasma, hot).')
@click.option('--figsize', type=(float, float), default=(10, 8), help='Figure size for the plot (width, height).')
def plot(h5_path, sample_id, rt_precision, mz_precision, output_path, cmap, figsize):
    """Plots the 2D mass spec image for a given sample ID from an HDF5 file."""
    click.echo(f"Plotting sample '{sample_id}' from {h5_path}...")
    try:
        plot_sample_image(
            h5_path=h5_path,
            sample_id=sample_id,
            target_rt_precision=rt_precision,
            target_mz_precision=mz_precision,
            output_path=output_path,
            cmap=cmap,
            figsize=figsize
        )
    except Exception as e:
        click.echo(click.style(f"An error occurred during plotting: {e}", fg='red'), err=True)

@main.command('align-rt')
@click.argument('h5_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('--metadata-csv', required=True, type=click.Path(exists=True, resolve_path=True),
              help='Metadata CSV for QC sample identification.')
@click.option('--qc-name-col', default='Sample Name', show_default=True,
              help='Column in metadata CSV with sample names.')
@click.option('--qc-type-col', default='sample_source', show_default=True,
              help='Column in metadata CSV with sample type.')
@click.option('--qc-type-val', default='QC', show_default=True,
              help='Value identifying QC samples in --qc-type-col.')
@click.option('--output', '-o', type=click.Path(writable=True, resolve_path=True),
              default=None,
              help='Output H5 path. If omitted, modifies the input file in-place.')
@click.option('--max-shift', default=30.0, type=float, show_default=True,
              help='Maximum RT shift to search for (seconds).')
@click.option('--segment-size', default=60.0, type=float, show_default=True,
              help='Cross-correlation segment size (seconds).')
def align_rt(h5_path, metadata_csv, qc_name_col, qc_type_col, qc_type_val,
             output, max_shift, segment_size):
    """Align retention times across samples using QC-based BPC cross-correlation.

    Reads the QC sample list from METADATA_CSV, computes per-sample RT shift
    corrections by cross-correlating each sample's Base Peak Chromatogram (BPC)
    against the median-TIC QC reference, then applies the corrections to
    rt_indices in H5_PATH.

    Without QC samples this command cannot function — all non-QC runs are
    corrected relative to the pooled-QC reference.
    """
    import pandas as pd
    from .alignment import align_rt as _align_rt

    meta = pd.read_csv(metadata_csv)

    if qc_name_col not in meta.columns:
        raise click.BadParameter(
            f"Column '{qc_name_col}' not found in {metadata_csv}. "
            f"Available columns: {list(meta.columns)}"
        )
    if qc_type_col not in meta.columns:
        raise click.BadParameter(
            f"Column '{qc_type_col}' not found in {metadata_csv}. "
            f"Available columns: {list(meta.columns)}"
        )

    qc_names = meta.loc[meta[qc_type_col] == qc_type_val, qc_name_col].dropna().tolist()
    if len(qc_names) == 0:
        raise click.UsageError(
            f"No QC samples found: column '{qc_type_col}' == '{qc_type_val}' "
            f"matched 0 rows in {metadata_csv}."
        )

    click.echo(f"Found {len(qc_names)} QC samples for RT alignment reference.")
    if output is None:
        click.echo(f"Modifying {h5_path} in-place.")
    else:
        click.echo(f"Writing corrected H5 to {output}.")

    try:
        _align_rt(
            h5_path=h5_path,
            qc_sample_names=qc_names,
            output_path=output,
            max_shift_s=max_shift,
            segment_size_s=segment_size,
        )
        click.echo(click.style("RT alignment complete.", fg='green'))
    except Exception as e:
        click.echo(click.style(f"RT alignment failed: {e}", fg='red'), err=True)
        raise SystemExit(1)


@main.command()
@click.argument('mzml_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('output_csv', type=click.Path(writable=True, resolve_path=True))
@click.option('--plot', type=click.Path(writable=True, resolve_path=True), default=None,
              help='Also save a precursor-vs-product m/z plot (PNG) to this path.')
def ms1ms2(mzml_path, output_csv, plot):
    """
    Analyzes a single mzML file to generate MS1 vs MS2 cumulative response (TIC) data.

    MZML_PATH: Path to the input mzML file.
    OUTPUT_CSV: Path to save the output CSV file.
    """
    try:
        analyze_ms1_ms2_response(mzml_path, output_csv)
        if plot:
            plot_ms1ms2_response(output_csv, plot)
    except Exception as e:
        click.echo(click.style(f"An error occurred: {e}", fg='red'), err=True)

if __name__ == '__main__':
    main()
