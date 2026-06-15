import torch
import numpy as np

from . import s2p_reg_utils as utils
from . import rigid, nonrigid



def compute_reference_image(
    frames,
    settings,
    device=torch.device("cpu"),
):
    """
    Compute the reference image by iterative rigid alignement.

    PARAMETERS
        frames: np.ndarray
            Frames of shape (nimg_init, Ly, Lx), dtype int 16

        settings: dict
            Settings dict that contains keys "batch size",
            "smooth_sigma", "spatial_taper", "maxregshift"

        device: torch.device
            Torch device (CPU or Cuda)

    OUTPUTS
        refImg: np.ndarray
            Reference image of shape (Ly, Lx), dtype int16
    
    """
    # Convert to torch tensor
    fr_reg = torch.from_numpy(frames)
    # Get the initial reference image
    refImg = pick_initial_reference(fr_reg)

    # Iterate to refine the reference image
    niter = 8
    batch_size = settings["batch_size"]

    for iter in range(0, niter):
        ## rigid registration shifts to initial reference
        maskMul, maskOffset, cfRefImg = compute_filters_and_norm(refImg, False, settings["smooth_sigma"],
                                           settings["spatial_taper"], block_size=None, device=device)[:3]
         
        # rigid registration in batches
        for k in range(0, fr_reg.shape[0], batch_size):
            fr_reg_batch = fr_reg[k:min(k + batch_size, fr_reg.shape[0])].to(device)
            ymax, xmax, cmax = rigid.phasecorr(fr_reg_batch, cfRefImg, maskMul, maskOffset,
                maxregshift=settings["maxregshift"],
                smooth_sigma_time=settings["smooth_sigma_time"])[:3]
            
            # shift frames to reference
            fr_reg_batch = torch.stack([torch.roll(frame, shifts=(-dy, -dx), dims=(0, 1))
                                for frame, dy, dx in zip(fr_reg_batch, ymax, xmax)], axis=0)
            # Update fr_reg
            fr_reg[k:min(k + batch_size, fr_reg.shape[0])] = fr_reg_batch.cpu()

        # Frames to average for new reference
        nmax = max(2, int(frames.shape[0] * (1. + iter) / (2 * niter)))
        isort = torch.argsort(-cmax)[:nmax].cpu()
        refImg = fr_reg[isort].double().mean(dim=0)

        # recenter reference image
        if device.type == 'mps':
            # MPS backend currently can not support float64
            dy, dx = -torch.round(ymax[isort].to(torch.float32).mean()).int(), -torch.round(xmax[isort].to(torch.float32).mean()).int()
        else:
            dy, dx = -torch.round(ymax[isort].double().mean()).int(), -torch.round(xmax[isort].double().mean()).int()
        
        refImg = torch.roll(refImg, shifts=(-dy, -dx), dims=(0, 1))
        refImg = refImg.numpy().astype("int16")


    del fr_reg_batch 
    if device.type == "cuda":
        torch.cuda.empty_cache()    
        torch.cuda.synchronize()

    if device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()

    return refImg


def compute_filters_and_norm(
    refImg,
    norm_frames=True,
    spatial_smooth=1.15,
    spatial_taper=3.45,
    block_size=(128, 128),
    lpad=3,
    subpixel=10,
    device=torch.device("cpu"),
):
    """
    Compute registration masks, smoothed reference FFTs, and normalization bounds

    PARAMETERS  
        refImg : np.ndarray or list of np.ndarray
            Reference image of shape (Ly, Lx), or a list of reference images for
            multi-plane registration.

        norm_frames : bool
            If True, clip the reference image to [1st, 99th] percentile and return
            the clipping bounds.

        spatial_smooth : float
            Standard deviation (in pixels) of Gaussian smoothing applied to the
            reference image in the frequency domain.

        spatial_taper : float
            Scalar controlling the slope of the sigmoid spatial taper mask at image
            borders.

        block_size : tuple of int or None
            Block size (Ly_block, Lx_block) for nonrigid registration. If None,
            nonrigid masks are not computed.

        lpad : int
            Number of pixels to pad each nonrigid block.

        subpixel : int
            Subpixel accuracy factor for nonrigid block shifts.

        device : torch.device
            Torch device to move the masks and reference FFTs to.

    OUTPUT
        tuple
            If refImg is a single image, returns (maskMul, maskOffset, cfRefImg,
            maskMulNR, maskOffsetNR, cfRefImgNR, blocks, rmin, rmax). If refImg is
            a list, returns a list of such tuples.
    
    """
    # Check for list of reference images
    if isinstance(refImg, list):
        refAndMasks_all = []
        for rimg in refImg:

            refAndMasks = compute_filters_and_norm(rimg, norm_frames=norm_frames, 
                                                   spatial_smooth=spatial_smooth, 
                                                   spatial_taper=spatial_taper, 
                                                   lpad=lpad, subpixel=subpixel, 
                                                   block_size=block_size, device=device)

            refAndMasks_all.append(refAndMasks)
        return refAndMasks_all
    
    else:
        # Normalize frames if specified
        if norm_frames:
            refImg, rmin, rmax = normalize_reference_image(refImg)
        else:
            rmin, rmax = -np.inf, np.inf

        # Convert to tensor
        rimg = torch.from_numpy(refImg)

        # compute the masks used for registration
        maskMul, maskOffset, cfRefImg = rigid.compute_masks_ref_smooth_fft(refImg=rimg, maskSlope=spatial_taper,
                                                                 smooth_sigma=spatial_smooth)
        
        Ly, Lx = refImg.shape
        # MPS backend does not support float64, convert to float32
        if device.type == "mps":
            maskMul, maskOffset = maskMul.to(torch.float32), maskOffset.to(torch.float32)
            cfRefImg = cfRefImg.to(torch.complex64)
        
        maskMul, maskOffset = maskMul.to(device), maskOffset.to(device)
        cfRefImg = cfRefImg.to(device)
        # Compute for blocks used in non-rigid registration
        blocks = []
        if block_size is not None:
            blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size,
                                          lpad=lpad, subpixel=subpixel)
            maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.compute_masks_ref_smooth_fft(
                refImg0=rimg, maskSlope=spatial_taper, smooth_sigma=spatial_smooth,
                yblock=blocks[0], xblock=blocks[1],
            )
            # MPS backend does not support float64, convert to float32
            if device.type == "mps":
                maskMulNR, maskOffsetNR = maskMulNR.to(torch.float32), maskOffsetNR.to(torch.float32)
                cfRefImgNR = cfRefImgNR.to(torch.complex64)
            maskMulNR, maskOffsetNR = maskMulNR.to(device), maskOffsetNR.to(device)
            cfRefImgNR = cfRefImgNR.to(device)

        else:
            maskMulNR, maskOffsetNR, cfRefImgNR = None, None, None

        return (maskMul, maskOffset, cfRefImg, 
                maskMulNR, maskOffsetNR, cfRefImgNR, 
                blocks,
                rmin, rmax)
    

def pick_initial_reference(frames):
    """
    Compute the initial reference image by finding the most correlated frame.

    The seed frame is the frame with the largest mean pairwise correlation with its
    top 20 correlated frame pairs. The initial reference is the average of that seed frame
    and its top 20 correlated frames

    PARAMETERS
        frames: torch.Tensor
            Input frames of shape (n_frames, Ly, Lx)

    OUTPUT
        refImg: np.ndarray
            Initial reference image of shape (Ly, Lx), dtype int16

    """
    nimg, Ly, Lx = frames.shape
    fr_z = frames.clone().reshape(nimg, -1).double()
    fr_z = fr_z.mean(dim=1, keepdim=True)
    cc = fr_z @ fr_z.T
    ndiag = torch.diag(cc) ** 0.5
    cc = cc / torch.outer(ndiag, ndiag)
    CCsort = -torch.sort(-cc, dim=1)[0]
    # Find frame most correlated to other frames
    bestCC = CCsort[:, 1:20].mean(dim=1)
    imax = torch.argmax(bestCC)
    # Average top 20 frames most correlated with imax
    indsort = torch.argsort(-cc[imax, :])
    refImg = fr_z[indsort[:20]].mean(axis=0).cpu().numpy().astype("int16")
    refImg = refImg.reshape(Ly, Lx)

    return refImg

def normalize_reference_image(refImg):
    """
    Clip reference image to [1st, 99th] intensity percentiles.

    PARAMETERS
        refImg : np.ndarray
            Reference image of shape (Ly, Lx).

    OUTPUT
        refImg : np.ndarray
            Clipped reference image of shape (Ly, Lx).

        rmin : np.int16
            1st percentile intensity value used as the lower clip bound.

        rmax : np.int16
            99th percentile intensity value used as the upper clip bound.
    """
    rmin, rmax = np.percentile(refImg, [1, 99]).astype(np.int16)
    refImg = np.clip(refImg, rmin, rmax)

    return refImg, rmin, rmax