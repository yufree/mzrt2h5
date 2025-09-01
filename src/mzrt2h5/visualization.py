import matplotlib.pyplot as plt
import numpy as np
import h5py
from .dataset import DynamicSparseH5Dataset

def plot_sample_image(
    h5_path: str,
    sample_id: str,
    target_rt_precision: float,
    target_mz_precision: float,
    output_path: str = None,
    cmap: str = 'viridis',
    figsize: tuple = (10, 8)
):
    """
    Plots the 2D mass spec image for a given sample ID from an HDF5 file.

    Args:
        h5_path (str): Path to the HDF5 file.
        sample_id (str): The sample ID (e.g., 'Sample_A') to plot.
        target_rt_precision (float): The RT precision to use for image reconstruction.
        target_mz_precision (float): The m/z precision to use for image reconstruction.
        output_path (str, optional): Path to save the plot. If None, displays the plot.
        cmap (str): Colormap for the plot.
        figsize (tuple): Figure size for the plot.
    """
    try:
        dataset = DynamicSparseH5Dataset(
            h5_path=h5_path,
            target_rt_precision=target_rt_precision,
            target_mz_precision=target_mz_precision,
            augment=False # No augmentation for visualization
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Find the sample index based on sample_id
    sample_idx = -1
    sample_name_found = False
    if 'Sample Name' in dataset.covariates:
        sample_names = [s.decode('utf-8') if isinstance(s, bytes) else s for s in dataset.covariates['Sample Name']]
        if sample_id in sample_names:
            sample_idx = sample_names.index(sample_id)
            sample_name_found = True
    
    if not sample_name_found:
        try:
            # If sample_id is not a name, try to interpret it as an integer index
            sample_idx = int(sample_id)
            if sample_idx < 0 or sample_idx >= len(dataset.filtered_indices):
                raise ValueError("Sample index out of bounds.")
        except ValueError:
            print(f"Error: Sample ID '{sample_id}' not found in 'Sample Name' covariate and is not a valid integer index.")
            return

    # Get the image tensor for the specified sample
    image_tensor, _ = dataset[sample_idx]
    image_np = image_tensor.squeeze().cpu().numpy() # Remove batch and channel dims

    plt.figure(figsize=figsize)
    plt.imshow(image_np, cmap=cmap, aspect='auto',
               extent=[dataset.storage_mz_range[0], dataset.storage_mz_range[1],
                       dataset.storage_rt_range[1], dataset.storage_rt_range[0]]) # RT is usually plotted from top to bottom
    plt.colorbar(label='Intensity')
    plt.xlabel('m/z')
    plt.ylabel('Retention Time (s)')
    plt.title(f'Mass Spec Image for Sample: {sample_id}')
    plt.grid(False)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()
