import numpy as np
import torch
import time

import logging

from .patchwarp_rigid import patchwarp_rigid
from ..io.assign_reg_io import assign_reg_io

logger = logging.getLogger(__name__)

from .. import parameters


def patchwarp_wrapper(
    f_raw_chan1,
    f_reg_chan1,
    f_raw_chan2,
    f_reg_chan2=None,
    refImg=None,
    align_by_chan2=False,
    save_path=None,
    aspect=1.0,
    settings=parameters.default_settings()["Patchwarp_settings"],
    save_tif=False,
    downsampled_frames=50,
    device=torch.device("cpu"),
):
    """
    Main registration function for patchwarp registration. For single and two channel images

    It will handle running the rigid and nonrigid motion correction for each imaging channel
    provided

    PARAMETERS
        f_raw_chan1: np.ndarray or BinaryFile
            Raw functional channel frames of shape (n_frmaes, Ly, Lx)

        f_reg_chan1: np.ndarray or BinaryFile
            Registered functional channel frames of shape (n_frmaes, Ly, Lx)

        f_raw_chan2: np.ndarray or BinaryFile
            Raw second channel

        f_reg_chan2: np.ndarray or BinaryFile
            Registered second channel

        refImg: np.ndarray or None
            Reference image of shape (Ly, Lx), dtype int16. If none, reference is computed
            from the data

        align_by_chan2: bool
            If True, use the second channel as the alignement source

        save_path: str or None
            Base directory for saving registered tiff files

        aspect: float
            Pixel aspect ratio used for computing the enhanced mean image

        settings: dict
            Registration settings dictionary

        device: torch.device
            Torch device from computation

    OUTPUTS
        reg_outputs : dict


    """

    # Get the input outputs for registration
    # Get the input outputs for registration
    f_out = assign_reg_io(
        f_raw_chan1,
        f_reg_chan1,
        f_raw_chan2,
        f_reg_chan2,
        align_by_chan2,
        save_path,
        save_tif,
    )
    # Unpack i/o
    f_align_in, f_align_out, f_alt_in, f_alt_out, tif_align_out, tif_alt_out = f_out

    nchannels = 2 if f_alt_in is not None else 1
    logger.info(f"Registering {nchannels} channels")

    # Check to run rigid registration
    if settings["do_rigid"]:
        logger.info(f"Performing rigid registration")
        t_rigid = time.time()
        rigid_outputs = patchwarp_rigid(
            f_align_in=f_align_in,
            f_align_out=f_align_out,
            f_alt_in=f_alt_in,
            f_alt_out=f_alt_out,
            n_ch=nchannels,
            rigid_norm_method=settings["rigid_norm_method"],
            rigid_norm_radius=settings["rigid_norm_radius"],
            rigid_template_block_size=settings["rigid_template_block_size"],
            rigid_template_threshold=settings["rigid_template_threshold"],
            rigid_template_tiffstack_num=settings["rigid_template_tiffstack_num"],
            rigid_template_center_frac=settings["rigid_template_center_frac"],
            rigid_template_fftdenoise=settings["rigid_template_fftdenoise"],
            device=torch.device("cpu"),
            tif_root_align=tif_align_out,
            tif_root_alt=tif_alt_out,
        )
        logger.info(
            "-- Rigid registration complete, %0.2f sec." % (time.time() - t_rigid)
        )
    else:
        logger.info(f"Skipping rigid regitration")
