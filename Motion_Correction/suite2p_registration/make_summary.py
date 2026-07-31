import os
from pathlib import Path

import numpy as np

from .register import save_tiff


def make_summary_video(reg_file, downsample_frames=50):
    """
    Convert registered binary file into a summary tif video
    that is downsampled for each viewing

    PARAMETERS
        reg_file : np.ndarray or BinaryFile
                Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.

        downsample_frame: int
            Number of frames to sum over to generate the summary
    """
    d0, d1, d2 = reg_file.shape
    n_full_bins = d0 // 50
    remainder = d0 % downsample_frames

    # Reshape and sum data
    summed_mov = np.empty(
        (n_full_bins + (1 if remainder else 0), d1, d2), dtype=reg_file.dtype
    )

    for i in range(n_full_bins):
        summed_mov[i] = reg_file[i * 50 : (i + 1) * 50].mean(axis=0)

    if remainder:
        summed_mov[-1] = reg_file[n_full_bins * 50 :].mean(axis=0)

    # Prepare the save name
    directory = Path(reg_file.filename).parent
    bname = Path(reg_file.filename).stem
    fname = os.path.join(directory, f"{bname}.tif")

    save_tiff(mov=summed_mov, fname=fname)
