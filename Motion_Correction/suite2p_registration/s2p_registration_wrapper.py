import time
import os
from warnings import warn
from tqdm import trange

import numpy as np
import torch

from .assign_reg_io import assign_reg_io
from . import bidiphase as bidi
from .compute_reference_image import compute_reference_image

import logging
logger = logging.getLogger(__name__)

from .. import parameters, logger

def s2p_registration_wrapper(
    f_raw_chan1,
    f_reg_chan1,
    f_raw_chan2,
    f_reg_chan2=None,
    refImg=None,
    align_by_chan2=False,
    save_path=None,
    aspect=1.0,
    settings=parameters.default_settings(),
    save_tif=False,
    device=torch.device("cpu"),
):
    """
    Main registration function for suite2p registration. For single and two channel images

    Computes reference image (if not provided), estimates bidirectional phase offset, 
    registers the primary channel, optionally performs two-step registration, applies shifts
    to alternate channel if present, and returns all registration outputs as dictionary

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
            Dictionary containing registration results with keys: "refImg", "rmin",
            "rmax", "meanImg", "yoff", "xoff", "corrXY", "yoff1", "xoff1",
            "corrXY1", "meanImg_chan2", "badframes", "badframes0", "yrange",
            "xrange", "bidiphase", "meanImgE", and optionally "zpos_registration",
            "cmax_registration", "meanImg_upsample", "mean_img_ups", 
            and "counts_ups".

    """
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
    f_align_in, f_align_out, f_alt_in, f_atl_out, tif_align_out, tif_alt_out = f_out

    nchannels = 2 if f_alt_in is not None else 1
    logger.info(f"Registering {nchannels} channels")

    if device.type == "mps":
        logger.warning("MPS device does not support float64, using float32 for registration. "
                       "If you encounter registration issues, try using cuda or cpu instead.")
        
    ############# Compute reference image and bidiphase shift ##################
    n_frames, Ly, Lx = f_align_in.shape
    badframes0 = np.zeros(n_frames, bool)
    badframes = None

    compute_bidi = settings["do_bidiphase"] and settings["bidiphase"] == 0
    # Grab frames
    if refImg is None or compute_bidi:
        ix_frames = np.linspace(0, n_frames, 1 + min(settings["nimg_init"], n_frames), dtype=int)[:-1]
        frames = f_align_in[ix_frames].copy()
    
    # Compute bidiphase shift
    if compute_bidi:
        bidiphase = bidi.compute(frames)
        logger.info("Estimated bidiphase offest from data: %d pixels" % bidiphase)
    else:
        bidiphase = settings["bidiphase"]

    if bidiphase !=0 and refImg is None:
        frames = bidi.shift(frames, bidiphase)
    
    # Compute the reference image
    if refImg is None:
        t0 = time.time()
        refImg = compute_reference_image(frames, settings=settings, device=device)
        logger.info("Reference frame, %0.2f sec." % (time.time() - t0))
    
    refImg_orig = refImg.copy()

    # Register frames
    for step in range(1 + (settings["two_step_registration"])):
        if step == 1:
            logger.info("Starting step 2 of two-step registration")
            logger.info("Making new reference image without badframes")
            nsamps = min(n_frames, settings["nimg_init"])
            inds = np.linspace(0, n_frames, 1 + nsamps).astype(np.int64)[:-1]
            inds = inds[~np.isin(inds, np.nonzero(badframes)[0])]
            refImg = f_align_out[inds].astype(np.float32).mean(axis=0)
            refImg_orig = refImg.copy()

        ######### Register frames to reference image ##############


