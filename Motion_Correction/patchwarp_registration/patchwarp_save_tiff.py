import os
from pathlib import Path

import numpy as np
from tifffile import TiffWriter


def patchwarp_save_tiff(f_out, tiff_root, frame_count):
    """Helper function to help save registered tif files"""
    basename = Path(f_out.filename).stem
    # Get batches
    start = 0
    end = f_out.shape[0]
    batch_idxs = [
        (i, min(i + frame_count, end)) for i in range(start, end, frame_count)
    ]

    # Itereate through chunks and save
    for n, idx in enumerate(batch_idxs):
        frames = f_out[idx[0] : idx[1]]
        fname = os.path.join(tiff_root, f"{basename}_{n+1:04d}.tif")
        save_tiff(mov=frames, fname=fname)


def save_tiff(mov: np.ndarray, fname: str) -> None:
    """
    Save image stack array to a tiff file.

    Parameters
    ----------
    mov : np.ndarray
        Image stack of shape (nimg, Ly, Lx) to save. Values are floored and
        cast to int16 before writing.
    fname : str
        Output tiff file path.
    """
    from tifffile import TiffWriter

    with TiffWriter(fname) as tif:
        for frame in np.floor(mov).astype(np.int16):
            tif.write(frame, contiguous=True)
