import time
import os
from warnings import warn
from tqdm import trange

import numpy as np
import torch

from . import s2p_reg_utils as utils
from .assign_reg_io import assign_reg_io
from . import bidiphase as bidi
from .compute_reference_image import compute_reference_image
from .register import register_frames, compute_crop, shift_frames_and_write
from .make_summary import make_summary_video

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
    f_align_in, f_align_out, f_alt_in, f_alt_out, tif_align_out, tif_alt_out = f_out

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
        outputs = register_frames(
            f_align_in,
            f_align_out=f_align_out,
            refImg=refImg,
            batch_size=settings["batch_size"],
            bidiphase=bidiphase,
            norm_frames=settings["norm_frames"],
            smooth_sigma=settings["smooth_sigma"],
            spatial_taper=settings["spatial_taper"],
            block_size=settings["block_size"],
            nonrigid=settings["nonrigid"],
            maxregshift=settings["maxregshift"],
            smooth_sigma_time=settings["smooth_sigma_time"],
            snr_thresh=settings["snr_thresh"],
            maxregshiftNR=settings["maxregshiftNR"],
            subpixel=settings["subpixel"],
            device=device,
            tif_root=tif_align_out,
            apply_shifts=True,
            upsample_meanImg=settings.get("upsample_meanImg", False)
        )

        # Upack results
        rmin, rmax, mean_img, offsets_all, blocks, mean_img_ups, counts_ups, meanImg_ups = outputs
        yoff, xoff, corrXY, yoff1, xoff1, corrXY1, zest, cmax_all = offsets_all

        # Compute value region and timepoints to exclude
        badframes, yrange, xrange = compute_crop(
            xoff=xoff,
            yoff=yoff,
            corrXY=corrXY,
            th_badframes=settings["th_badframes"],
            badframes=badframes0.copy(),
            maxregshift=settings["maxregshift"],
            Ly=Ly,
            Lx=Lx,
        )

    
    ########### Register Second Channel ################
    if nchannels > 1:
        mean_img_alt = shift_frames_and_write(
            f_alt_in=f_alt_in,
            f_alt_out=f_alt_out,
            batch_size=settings["batch_size"],
            yoff=yoff,
            xoff=xoff,
            yoff1=yoff1,
            xoff1=xoff1,
            blocks=blocks,
            bidiphase=bidiphase,
            tif_root=tif_alt_out,
            device=device,
        )
    else:
        mean_img_alt = None

    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    if device.type == "mps":
        torch.mps.empty_cache()

    meanImg = mean_img if nchannels == 1 or not align_by_chan2 else mean_img_alt
    if nchannels == 2:
        meanImg_chan2 = mean_img_alt if not align_by_chan2 else mean_img
    else:
        meanImg_chan2 = None
    
    # Make and save summary video
    make_summary_video(f_align_out, downsample_frames=settings["downsampled_frames"])
    if nchannels > 1:
        make_summary_video(f_alt_out, downsample_frames=settings["downsample_frames"])

    # Package results for output
    reg_outputs = registration_outputs_to_dict(
        refImg_orig,
        rmin, 
        rmax,
        meanImg,
        (yoff, xoff, corrXY),
        (yoff1, xoff1, corrXY1),
        meanImg_chan2,
        badframes,
        badframes0,
        yrange,
        xrange,
        bidiphase,
    )

    # add enhanced mean image
    meanImgE = utils.highpass_mean_image(meanImg.astype("float32"), aspect=aspect)
    reg_outputs["meanImgE"] = meanImgE

    # add upsampled mean image if computed
    if mean_img_ups is not None and counts_ups is not None:
        reg_outputs["meanImg_upsample"] = meanImg_ups
        reg_outputs["mean_img_ups"] = mean_img_ups.cpu().numpy()
        reg_outputs["counts_ups"] = counts_ups.cpu().numpy()

    return reg_outputs


def registration_outputs_to_dict(
    refImg,
    rmin, 
    rmax,
    meanImg,
    rigid_offsets,
    nonrigid_offsets,
    meanImg_chan2,
    badframes,
    badframes0,
    yrange,
    xrange, 
    bidiphase,
):
    """
    Pack registration results into dictionary

    PARAMETERS
        refImg : np.ndarray
            Reference image of shape (Ly, Lx).

        rmin : np.int16
            Lower intensity clip bound.

        rmax : np.int16
            Upper intensity clip bound.

        meanImg : np.ndarray
            Mean registered image of shape (Ly, Lx).

        rigid_offsets : tuple
            Tuple of (yoff, xoff, corrXY) rigid registration offsets.

        nonrigid_offsets : tuple
            Tuple of (yoff1, xoff1, corrXY1) nonrigid offsets, elements may be None.

        meanImg_chan2 : np.ndarray or None
            Mean image of the second channel, shape (Ly, Lx).

        badframes : np.ndarray
            1-D boolean array of detected bad frames.

        badframes0 : np.ndarray
            1-D boolean array of initial bad frames before registration.

        yrange : list of int
            [ymin, ymax] valid row range.

        xrange : list of int
            [xmin, xmax] valid column range.

        bidiphase : int
            Bidirectional phase offset in pixels.


    OUTPUT
        reg_outputs : dict
            Dictionary with keys "refImg", "rmin", "rmax", "yoff", "xoff",
            "corrXY", "meanImg", "badframes", "badframes0", "yrange", "xrange",
            "bidiphase", and optionally "yoff1", "xoff1", "corrXY1",
            "meanImg_chan2", "zpos_registration", "cmax_registration".
    
    """
    # Initialize output
    reg_outputs = {}

    # Assign reference image and normalizers
    reg_outputs["refImg"] = refImg
    reg_outputs["rmin"], reg_outputs["rmax"] = rmin, rmax

    # assign offsets
    reg_outputs["yoff"], reg_outputs["xoff"], reg_outputs["corrXY"] = rigid_offsets
    if nonrigid_offsets[0] is not None:
        reg_outputs["yoff1"], reg_outputs["xoff1"], reg_outputs["corrXY1"] = nonrigid_offsets
    
    # Assign mean images
    reg_outputs["meanImg"] = meanImg
    if meanImg_chan2 is not None:
        reg_outputs["meanImg_chan2"] = meanImg_chan2
    
    # Assign crop computation and bad frames
    reg_outputs["badframes"], reg_outputs["badframes0"] = badframes, badframes0
    reg_outputs["yrange"], reg_outputs["xrange"] = yrange, xrange

    reg_outputs["bidiphase"] = bidiphase

    return reg_outputs


