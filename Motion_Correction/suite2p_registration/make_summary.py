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
    data = reg_file.copy()
    d0, d1, d2 = data.shape
    remainder = d0 % downsample_frames

    if remainder != 0:
        data = data[:-remainder, :, :]
    
    # Reshape and sum data
    summed_mov = data.reshape(d0 // downsample_frames, downsample_frames, d1, d2).sum(axis=1)

    # Prepare the save name
    directory = Path(reg_file.filename).parent
    bname = Path(reg_file.filename).stem
    fname = os.path.join(directory, f"{bname}.tif")

    save_tiff(mov=summed_mov, fname=fname)

