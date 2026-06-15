import os
from tqdm import trange
from pathlib import Path
from warnings import warn

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt

from ..logger import TqdmToLogger

from . import rigid, nonrigid
from .compute_reference_image import compute_filters_and_norm
from . import bidiphase as bidi

import logging

logger = logging.getLogger(__name__)


def register_frames(
    f_align_in,
    refImg,
    f_align_out, 
    batch_size=100,
    bidiphase=0,
    norm_frames=True,
    smooth_sigma=1.15,
    spatial_taper=3.45,
    block_size=(128,128),
    nonrigid=True,
    maxregshift=0.1,
    smooth_sigma_time=0,
    snr_thresh=1.2,
    maxregshiftNR=5,
    subpixel=10,
    device=torch.device("cpu"),
    tif_root=None,
    apply_shifts=True,
    upsample_meanImg=False,
):
    """
    Register frames to a reference image using rigid and optionally nonrigid shifts.

    Computes registration masks from the reference, then processes frames in
    batches: computes shifts, applies them, accumulates a mean image, and
    optionally writes registered frames to f_align_out. Supports multi-plane
    registration when refImg is a list.

    PARAMETERS
        f_align_in : np.ndarray or BinaryFile
            Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.

        refImg : np.ndarray or list of np.ndarray
            Reference image of shape (Ly, Lx), or a list for multi-plane registration.

        f_align_out : np.ndarray or BinaryFile or None
            Output array for registered frames. If None, registered frames are
            written back to f_align_in.

        batch_size : int
            Number of frames to process per batch.

        bidiphase : int
            Bidirectional phase offset in pixels. If non-zero, frames are corrected
            before registration.

        norm_frames : bool
            If True, clip frames to the reference image's [1st, 99th] percentile range.

        smooth_sigma : float
            Standard deviation of Gaussian smoothing applied to the reference image.

        spatial_taper : float
            Slope of the sigmoid spatial taper mask at image borders.

        block_size : tuple of int
            Block size (Ly_block, Lx_block) for nonrigid registration.

        nonrigid : bool
            If True, compute nonrigid shifts in addition to rigid shifts.

        maxregshift : float
            Maximum rigid shift as a fraction of the smaller image dimension.

        smooth_sigma_time : float
            Sigma for temporal smoothing of phase-correlation maps.

        snr_thresh : float
            SNR threshold for accepting nonrigid block shifts.

        maxregshiftNR : int
            Maximum nonrigid shift in pixels.

        device : torch.device
            Torch device for computation.

        tif_root : str or None
            If provided, save registered frames as tiffs in this directory.

        apply_shifts : bool
            If True, apply computed shifts to frames. If False, only compute shifts.

        upsample_meanImg : bool, int, list, or tuple
            Upsampling factor for super-resolution mean image computation.
            If False or None, no upsampling is performed. If int, same factor is used
            for both Y and X. If list/tuple of length 2, specifies [Y_factor, X_factor].
            The mean image is computed by accumulating registered frames at subpixel
            locations and normalizing by pixel counts.

    OUTPUT
        rmin : np.int16 or list
            Lower intensity clip bound(s) from reference normalization.

        rmax : np.int16 or list
            Upper intensity clip bound(s) from reference normalization.

        mean_img : np.ndarray
            Mean registered image of shape (Ly, Lx).

        offsets_all : list
            List of [yoff, xoff, corrXY, yoff1, xoff1, corrXY1, zest, cmax_all]
            concatenated across all batches.

        blocks : list
            Block definitions from nonrigid.make_blocks.

        mean_img_ups : torch.Tensor or None
            Raw upsampled mean image tensor of shape (Ly*upsample[0], Lx*upsample[1])
            before normalization. None if upsample_meanImg is False.

        counts_ups : torch.Tensor or None
            Pixel counts tensor of shape (Ly*upsample[0], Lx*upsample[1]) indicating
            how many frames contributed to each upsampled pixel. None if upsample_meanImg is False.

        meanImg_ups : np.ndarray or None
            Super-resolution mean image of shape (Ly*upsample[0], Lx*upsample[1])
            after Gaussian smoothing and normalization by counts. None if upsample_meanImg is False.
    
    """
    n_frames, Ly, Lx = f_align_in.shape

    refAndMasks = compute_filters_and_norm(refImg, norm_frames=norm_frames,
                                           spatial_smooth=smooth_sigma,
                                           spatial_taper=spatial_taper,
                                           block_size=block_size if nonrigid else None,
                                           subpixel=subpixel, device=device)
    
    blocks = refAndMasks[-3] 
    rmin = refAndMasks[-2] 
    rmax = refAndMasks[-1] 

    ################### Register frames to reference image ########################
    mean_img = np.zeros((Ly, Lx), "float32")
    if upsample_meanImg:
        if not isinstance(upsample_meanImg, (np.ndarray, list, tuple)):
            upsample_meanImg = [upsample_meanImg, upsample_meanImg]
        
        ups_dtype = torch.float32 if device.type == "mps" else torch.double
        mean_img_ups = torch.zeros((int(Ly*upsample_meanImg[0]), int(Lx*upsample_meanImg[1])), dtype=ups_dtype, device=device)
        counts_ups = torch.zeros((int(Ly*upsample_meanImg[0]), int(Lx*upsample_meanImg[1])), dtype=torch.int, device=device)
    else:
        mean_img_ups, counts_ups, meanImg_ups = None, None, None

    # Determine number of batches
    n_batches = int(np.ceil(n_frames / batch_size))
    logger.info(f"Registering {n_frames} frames in {n_batches} batches")
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    for n in trange(n_batches, mininterval=10, file=tqdm_out):
        ## Set frame range for the batch
        tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
        # Read data and convert to tensor
        frames = f_align_in[tstart : tend]
        if device.type == "cuda":
            fr_torch = torch.from_numpy(frames).pin_memory().to(device)
        else:
            fr_torch = torch.from_numpy(frames).to(device)
        if bidiphase != 0:
            fr_torch = bidi.shift(fr_torch, bidiphase)

        fr_reg = fr_torch.clone()
        # Calcualte the shifts from reference
        offsets = compute_shifts(
            refAndMasks,
            fr_reg,
            maxregshift=maxregshift,
            smooth_sigma_time=smooth_sigma_time,
            snr_thresh=snr_thresh,
            maxregshiftNR=maxregshiftNR,
        )
        # Unpack
        ymax, xmax, cmax, ymax1, xmax1, cmax1, zest, cmax_all = offsets

        # Apply shifts to image
        if apply_shifts:
            frames = shift_frames(
                fr_torch, ymax, xmax, ymax1, xmax1, blocks, mean_img_ups=mean_img_ups,
                counts_up=counts_ups, device=device,
            )

        # Convert to numpy and concatenate offsets
        ymax, xmax, cmax = ymax.cpu().numpy(), xmax.cpu().numpy(), cmax.cpu().numpy()

        if ymax1 is not None:
            ymax1, xmax1, cmax1 = ymax1.cpu().numpy(), xmax1.cpu().numpy(), cmax1.cpu().numpy()

        offsets = [ymax, xmax, cmax, ymax1, xmax1, cmax1, zest, cmax_all]
        offsets_all = ([np.concatenate((offset_all, offset), axis=0)
                        if offset is not None else None
                        for offset_all, offset in zip(offsets_all, offsets)]
                        if n > 0 else offsets)

        # Make mean image from all registered frames
        mean_img += frames.sum(axis=0) / n_frames

        # Save aligned frames to bin file
        if apply_shifts:
            f_align_out[tstart : tend] = frames

            # Save aligned image as tif if specified
            if tif_root:
                bname = Path(f_align_out).stem
                fname = os.path.join(tif_root, f"{bname}_{n : 0.5d}.tif")
                save_tiff(mov=frames, fname=fname)
    
    if upsample_meanImg:
        # apply Gaussian smoothing and normalize
        mimg = mean_img_ups.cpu().numpy()
        cimg = counts_ups.cpu().numpy()
        sig = 1
        mimg = gaussian_filter(mimg, sig)
        cimg = gaussian_filter(cimg, sig)
        meanImg_ups = mimg / cimg

    return rmin, rmax, mean_img, offsets_all, blocks, mean_img_ups, counts_ups, meanImg_ups



def shift_frames_and_write(
    f_alt_in, 
    f_alt_out=None, 
    batch_size=100, 
    yoff=None, 
    xoff=None, 
    yoff1=None,
    xoff1=None, 
    blocks=None, 
    bidiphase=0, 
    device=torch.device("cuda"), 
    tif_root=None
):
    """
    Apply pre-computed registration shifts to an alternate channel and write results.

    Applies rigid (and optionally nonrigid) shifts that were computed on the
    primary channel to the alternate channel frames, in batches. Writes the
    shifted frames to f_alt_out if provided, otherwise overwrites f_alt_in.

    PARAMETERS  
        f_alt_in : np.ndarray or BinaryFile
            Alternate channel input frames of shape (n_frames, Ly, Lx).

        f_alt_out : np.ndarray or BinaryFile or None
            Output array for shifted frames. If None, writes back to f_alt_in.

        batch_size : int
            Number of frames per batch.

        yoff : np.ndarray
            Rigid y offsets of length n_frames.

        xoff : np.ndarray
            Rigid x offsets of length n_frames.

        yoff1 : np.ndarray or None
            Nonrigid y offsets of shape (n_frames, n_blocks).

        xoff1 : np.ndarray or None
            Nonrigid x offsets of shape (n_frames, n_blocks).

        blocks : list or None
            Block definitions from nonrigid.make_blocks.

        bidiphase : int
            Bidirectional phase offset in pixels.

        device : torch.device
            Torch device for computation.

        tif_root : str or None
            If provided, save shifted frames as tiffs in this directory.

    OUTPUT
    mean_img : np.ndarray
        Mean image of the shifted alternate channel, shape (Ly, Lx).
    """
    n_frames, Ly, Lx = f_alt_in.shape
    check_offsets(yoff, xoff, yoff1, xoff1, n_frames)

    mean_img = np.zeros((Ly, Lx), "float32")
    yoff1k, xoff1k = None, None

    n_batches = int(np.ceil(n_frames / batch_size))
    logger.info(f"Second channel: Shifting {n_frames} frames in {n_batches} batches")
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    # Iterate through the batched
    for n in trange(n_batches, mininterval=10, file=tqdm_out):
        tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
        # Read in the data to shift
        frames = f_alt_in[tstart : tend]
        yoffk, xoffk = yoff[tstart : tend].astype(int), xoff[tstart : tend].astype(int)

        if yoff1 is not None:
            yoff1k, xoff1k = yoff1[tstart : tend], xoff1[tstart : tend]
            
        if device.type == "cuda":
            fr_torch = torch.from_numpy(frames).pin_memory().to(device)
        else:
            fr_torch = torch.from_numpy(frames).to(device)

        if bidiphase != 0:
            fr_torch = bidi.shift(fr_torch, bidiphase)
        # Shift the frames
        frames = shift_frames(fr_torch, yoffk, xoffk, yoff1k, xoff1k, blocks, device=device)
        mean_img += frames.sum(axis=0) / n_frames

        f_alt_out[tstart : tend] = frames

        # save aligned frames to tiffs
        if tif_root:
            bname = Path(f_alt_out).stem
            fname = os.path.join(tif_root, f"{bname}_{n : 0.5d}.tif")
            save_tiff(mov=frames, fname=fname)

    return mean_img



def compute_shifts(
    refAndMasks,
    fr_reg,
    maxregshift=0.1,
    smooth_sigma_time=0,
    snr_thresh=1.2,
    maxregshiftNR=5,
):
    """
    Compute rigid and nonrigid registration shifts for a batch of frames.

    Performs rigid phase-correlation registration, then (if nonrigid masks are
    provided) applies rigid shifts and computes nonrigid block shifts. For
    multi-plane data (nZ > 1), selects the best z-plane per frame by maximum
    correlation.

    PARAMETERS
        refAndMasks : tuple or list of tuple
            Registration masks and reference FFTs from compute_filters_and_norm. If
            nZ > 1, a list of tuples (one per z-plane).

        fr_reg : torch.Tensor
            Frames to register, shape (N, Ly, Lx).

        maxregshift : float
            Maximum allowed rigid shift as a fraction of the smaller image dimension.

        smooth_sigma_time : float
            Sigma for temporal smoothing of phase-correlation maps. If <= 0, no
            temporal smoothing is applied.

        snr_thresh : float
            Signal-to-noise ratio threshold for accepting nonrigid block shifts.

        maxregshiftNR : int
            Maximum allowed nonrigid shift in pixels.

    OUTPUT
        ymax : torch.LongTensor
            1-D rigid y shifts of length N.

        xmax : torch.LongTensor
            1-D rigid x shifts of length N.

        cmax : torch.Tensor
            1-D rigid correlation values of length N.

        ymax1 : torch.Tensor or None
            Nonrigid y shifts of shape (N, n_blocks), or None if nonrigid is disabled.

        xmax1 : torch.Tensor or None
            Nonrigid x shifts of shape (N, n_blocks), or None if nonrigid is disabled.

        cmax1 : torch.Tensor or None
            Nonrigid correlation values of shape (N, n_blocks), or None.


    """

    (maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, 
        blocks, rmin, rmax) = refAndMasks
    
    device = fr_reg.device

    fr_reg = torch.clip(fr_reg, rmin, rmax) if rmin > -np.inf else fr_reg

    # rigid registration
    ymax, xmax, cmax = rigid.phasecorr(fr_reg, cfRefImg, maskMul, maskOffset, 
                                    maxregshift, smooth_sigma_time)[:3]
    
    # nonrigid registration
    if maskMulNR is not None and maxregshiftNR > 0:
        # Shift torch frames to the reference
        fr_reg = torch.stack([torch.roll(frame, shifts=(-dy, -dx), dims=(0,1))
                              for frame, dy, dx in zip(fr_reg, ymax, xmax)], axis=0)
        ymax1, xmax1, cmax1 = nonrigid.phasecorr(fr_reg, blocks,
                                                 maskMulNR, maskOffsetNR, cfRefImgNR,
                                                 snr_thresh, maxregshiftNR)[:3]
    else:
        ymax1, xmax1, cmax1 = None, None, None

    del fr_reg
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()
    
    return ymax, xmax, cmax, ymax1, xmax1, cmax1, None, None


def shift_frames(
    fr_torch,
    yoff,
    xoff,
    yoff1=None,
    xoff1=None,
    blocks=None,
    mean_img_ups=None,
    counts_up=None,
    device=torch.device("cpu"),
):
    """
    Apply rigid and optionally nonrigid shifts to frames and return as numpy int16.

    PARAMETERS
        fr_torch : torch.Tensor
            Frames to shift, shape (N, Ly, Lx).

        yoff : torch.LongTensor
            1-D rigid y shifts of length N.

        xoff : torch.LongTensor
            1-D rigid x shifts of length N.

        yoff1 : torch.Tensor or np.ndarray or None
            Nonrigid y shifts of shape (N, n_blocks). If None, only rigid shifts are
            applied.

        xoff1 : torch.Tensor or np.ndarray or None
            Nonrigid x shifts of shape (N, n_blocks).

        blocks : list or None
            Block definitions from nonrigid.make_blocks, used for nonrigid
            interpolation.

        device : torch.device
            Torch device for nonrigid shift tensors.

    OUTPUT
        frames_out : np.ndarray
            Shifted frames of shape (N, Ly, Lx), dtype matching the torch output.
    
    
    """
    # Perform the rigid shifts
    fr_torch = torch.stack([torch.roll(frame, shifts=(-dy, -dx), dims=(0, 1))
                               for frame, dy, dx in zip(fr_torch, yoff, xoff)], axis=0)
    
    # Non rigid shifts
    if yoff1 is not None:
        if isinstance(yoff1, np.ndarray):
            if fr_torch.device.type == "cuda":
                yoff1 = torch.from_numpy(yoff1).pin_memory().to(device)
                xoff1 = torch.from_numpy(xoff1).pin_memory().to(device)
            elif device.type == "mps":
                # MPS backend does not support float64
                yoff1 = torch.from_numpy(yoff1).to(torch.float32).to(device)
                xoff1 = torch.from_numpy(xoff1).to(torch.float32).to(device)
            else:
                yoff1 = torch.from_numpy(yoff1).to(device)
                xoff1 = torch.from_numpy(xoff1).to(device)
        
        fr_torch = nonrigid.transform_data(fr_torch, blocks[2], blocks[1], blocks[0],
                                           yoff1, xoff1, data_ups=mean_img_ups, counts_ups=counts_up)
    
    frames_out = np.empty(fr_torch.shape, dtype="int16")
    frames_out = fr_torch.cpu().numpy()

    return frames_out

def compute_crop(
        xoff, 
        yoff, 
        corrXY, 
        th_badframes, 
        badframes, 
        maxregshift,
        Ly, 
        Lx
):
    """
    Determine how much to crop the FOV based on registration motion offsets.

    Identifies badframes (frames with large outlier shifts, thresholded by
    th_badframes) and excludes them when computing valid y and x ranges for
    cropping the field of view.

    PARAMETERS
        xoff : np.ndarray
            1-D array of length n_frames with x (column) rigid registration offsets.

        yoff : np.ndarray
            1-D array of length n_frames with y (row) rigid registration offsets.

        corrXY : np.ndarray
            1-D array of length n_frames with phase-correlation values for each frame.

        th_badframes : float
            Threshold multiplier for detecting bad frames based on the ratio of shift
            deviation to correlation quality.

        badframes : np.ndarray
            1-D boolean array of length n_frames with pre-existing bad frame labels.

        maxregshift : float
            Maximum allowed registration shift as a fraction of the image dimension.
            Frames exceeding 95% of this limit are marked as bad.

        Ly : int
            Height of a frame in pixels.

        Lx : int
            Width of a frame in pixels.

    OUTPUT
        badframes : np.ndarray
            Updated 1-D boolean array of length n_frames indicating bad frames.

        yrange : list of int
            [ymin, ymax] valid row range after cropping for motion.

        xrange : list of int
            [xmin, xmax] valid column range after cropping for motion.
    """
    filter_window = min((len(yoff) // 2) * 2 - 1, 101)
    dx = xoff - medfilt(xoff, filter_window)
    dy = yoff - medfilt(yoff, filter_window)
    # offset in x and y (normed by mean offset)
    dxy = (dx**2 + dy**2)**.5
    dxy = dxy / dxy.mean()
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = corrXY / medfilt(corrXY, filter_window)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    badframes = np.logical_or(px > th_badframes * 100, badframes)
    badframes = np.logical_or(abs(xoff) > (maxregshift * Lx * 0.95), badframes)
    badframes = np.logical_or(abs(yoff) > (maxregshift * Ly * 0.95), badframes)
    if badframes.mean() < 0.5:
        ymin = np.ceil(np.abs(yoff[np.logical_not(badframes)]).max())
        xmin = np.ceil(np.abs(xoff[np.logical_not(badframes)]).max())
    else:
        warn(
            "WARNING: >50% of frames have large movements, registration likely problematic"
        )
        ymin = np.ceil(np.abs(yoff).max())
        xmin = np.ceil(np.abs(xoff).max())
    ymax = Ly - ymin
    xmax = Lx - xmin
    yrange = [int(ymin), int(ymax)]
    xrange = [int(xmin), int(xmax)]

    return badframes, yrange, xrange


def check_offsets(yoff, xoff, yoff1, xoff1, n_frames):
    """
    Validate that registration offset arrays have the expected number of frames.

    Parameters
    ----------
    yoff : np.ndarray or None
        Rigid y offsets of length n_frames.
    xoff : np.ndarray or None
        Rigid x offsets of length n_frames.
    yoff1 : np.ndarray or None
        Nonrigid y offsets of shape (n_frames, n_blocks), or None.
    xoff1 : np.ndarray or None
        Nonrigid x offsets of shape (n_frames, n_blocks), or None.
    n_frames : int
        Expected number of frames.

    Raises
    ------
    ValueError
        If rigid offsets are None or any offset array length does not match
        n_frames.
    """
    if yoff is None or xoff is None:
        raise ValueError("no rigid registration offsets provided")
    elif yoff.shape[0] != n_frames or xoff.shape[0] != n_frames:
        raise ValueError(
            "rigid registration offsets are not the same size as input frames")
    if yoff1 is not None and (yoff1.shape[0] != n_frames or xoff1.shape[0] != n_frames):
        raise ValueError(
                "nonrigid registration offsets are not the same size as input frames")


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