import os

import torch
import numpy as np
import logging

from . import parameters
from .suite2p_registration import s2p_registration_wrapper

logger = logging.getLogger(__name__)

def pipeline(
    save_path,
    f_reg_chan1,
    f_raw_chan1,
    f_reg_chan2=None,
    f_raw_chan2=None,
    algorithm="Suite2p",
    settings=parameters.default_settings(),
    device=torch.device("cpu")
):
    """
    Pipeline to run motion correction on array or binary file

    PARAMETERS
        save_path: str
            Path to save results
        
        f_reg_chan1: np.ndarray or BinaryFile
            Registered frames, shape (n_frames, Ly, Lx)
        
        f_raw_chan1: np.ndarray or BinaryFile
            Unregistered frames, shape (n_frames, Ly, Lx)
        
        f_reg_chan2: np.ndarray or BinaryFile
            Registered frames for channel 2, shape (n_frames, Ly, Lx)
        
        f_raw_chan2: np.ndarray or BinaryFile
            Unregistered frames for channel 2, shape (n_frames, Ly, Lx)
        
        algorithm: str
            Specifies to perform Suite2p or Patchwarp motion correction
        
        settings: dict
            Dictionary of pipeline settings

        device: torch.device
            Torch device for performing operations

    OUTPUT
        reg_outputs: dict
            Registration outputs including shifts and reference image
    
    """
    # Determine the algorithm to run
    logger.info(f"NOTE: Running {algorithm} motion correction")

    if algorithm == "Suite2p":
        align_by_chan2 = settings["suite2p_settings"]["align_by_chan2"]
        reg_outputs = s2p_registration_wrapper(
            f_raw_chan1=f_raw_chan1,
            f_reg_chan1=f_reg_chan1,
            f_raw_chan2=f_raw_chan2,
            f_reg_chan2=f_reg_chan2,
            refImg=None,
            align_by_chan2=align_by_chan2,
            save_path=save_path,
            settings=settings["suite2p_settings"],
            save_tif=settings["save_tif"],
            device=device,
        )

        # Can add registration metrics later from suite2p

        # Save outputs
        np.save(os.path.join(save_path, "reg_outputs.npy"), reg_outputs)

    elif algorithm == "PatchWarp":
        raise ValueError("Currently only works with Suite2p registration")
    

    return reg_outputs
