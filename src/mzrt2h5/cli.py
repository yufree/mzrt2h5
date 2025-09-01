import click
from .processing import save_dataset_as_sparse_h5
from .visualization import plot_sample_image

@click.group()
def main():
    """A command-line tool for processing mzML files and visualizing mass spec data."""
    pass

@main.command()
@click.argument('folder', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument('save_path', type=click.Path(writable=True, resolve_path=True))
@click.option('--rt-precision', default=0.1, type=float, help='Bin size for the retention time axis.')
@click.option('--mz-precision', default=0.01, type=float, help='Bin size for the m/z axis.')
@click.option('--metadata-csv-path', type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True, help='Path to the metadata CSV file.')
@click.option('--sample-id-col', default='Sample Name', help='Column name for sample IDs in the metadata.')
@click.option('--separator', default=',', help='Separator for the metadata file.')
@click.option('--mz-range', type=(float, float), help='Fixed (min, max) m/z range.')
@click.option('--rt-range', type=(float, float), help='Fixed (min, max) RT range.')
def process(folder, save_path, rt_precision, mz_precision, metadata_csv_path, sample_id_col, separator, mz_range, rt_range):
    """Processes a folder of mzML files and saves them as a single, consolidated sparse HDF5 file."""
    
    # Click returns empty tuples for non-provided tuple options, convert to None
    mz_range = mz_range if mz_range else None
    rt_range = rt_range if rt_range else None

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
            mz_range=mz_range,
            rt_range=rt_range
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

if __name__ == '__main__':
    main()
