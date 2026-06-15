import numpy as np
from numpy import fft
from scipy.fftpack import next_fast_len
import torch
import torch.nn.functional as F

from .s2p_reg_utils import spatial_taper, kernelD2, mat_upsample, convolve, ref_smooth_fft


def calculate_nblocks(L: int, block_size: int):
    """
    Returns block_size and nblocks from dimension length and desired block size

    PARAMETERS
        L: int
            Number of pixels in one dimension in image.

        block_size: int
            Block size in pixels.

    OUTPUT
        block_size: int
            min(L, block_size).

        nblocks: int
            Number of blocks to make along dimension.
    """

    return (L, 1) if block_size >= L else (block_size,
                                           int(np.ceil(1.5 * L / block_size)))



def make_blocks(Ly, Lx, block_size, lpad=3, subpixel=10):
    """
    Compute overlapping registration blocks covering a 2D field of view.
    This function splits a full-frame image of size (Ly, Lx) into an array of
    overlapping rectangular blocks to be processed independently for nonrigid
    registration. Block start positions are computed so that blocks tile the image
    with (approximately) equal spacing and specified overlap determined by the
    requested block_size. The function also computes a spatial smoothing matrix
    (NRsm) over the block grid and an upsampling convolution matrix (Kmat) used
    for subpixel shift estimation.

    PARAMETERS
        Ly : int
            Number of pixels in the vertical dimension (image height).

        Lx : int
            Number of pixels in the horizontal dimension (image width).

        block_size : tuple[int, int]
            Block size in pixels as (block_height, block_width).

        lpad : int, optional
            Padding in pixels used when constructing the upsampling matrix.
            Passed to mat_upsample(...). Default is 3.

        subpixel : int, optional
            Subpixel upsampling factor. Passed to mat_upsample(...). Default is 10.
    
    OUTPUT
    yblock : list[numpy.ndarray]
        List of length (ny * nx) giving the vertical (row) slice for each block.
        Each element is a 1D integer numpy array [y_start, y_end] specifying the
        inclusive start (y_start) and exclusive end (y_end) indices of the block
        along the vertical axis. Blocks are ordered row-major by block-grid row
        (iy) then column (ix): block_idx = iy * nx + ix.

    xblock : list[numpy.ndarray]
        List of length (ny * nx) giving the horizontal (column) slice for each
        block. Each element is a 1D integer numpy array [x_start, x_end]
        specifying the inclusive start and exclusive end indices along the
        horizontal axis. Ordering matches yblock (row-major block-grid order).

    nblocks : list[int, int]
        Two-element list [ny, nx] with the number of blocks in the vertical and
        horizontal directions respectively (ny = number of block rows,
        nx = number of block columns).

    block_size : tuple[int, int]
        Effective block size used, min of input block size and frame size.

    NRsm : numpy.ndarray
        2D smoothing kernel matrix defined on the block grid. Shape is (ny, nx).
        This matrix (derived from kernelD2 over block grid coordinates) is
        used to smooth or regularize blockwise motion estimates spatially.

    Kmat : numpy.ndarray
        Upsampling kriging interpolation matrix returned by mat_upsample(lpad, subpixel).
        This matrix is used for subpixel shift estimation within +/- lpad pixels.

    nup : int
        Kmat.shape[-1].
   
    """
    block_size = (int(block_size[0]), int(block_size[1]))
    block_size_y, ny = calculate_nblocks(L=Ly, block_size=block_size[0])
    block_size_x, nx = calculate_nblocks(L=Lx, block_size=block_size[1])
    block_size = (block_size_y, block_size_x)

    # todo: could rounding to int here over-represent some pixels over others?
    ystart = np.linspace(0, Ly - block_size[0], ny).astype("int")
    xstart = np.linspace(0, Lx - block_size[1], nx).astype("int")
    yblock = [
        np.array([ystart[iy], ystart[iy] + block_size[0]])
        for iy in range(ny)
        for _ in range(nx)
    ]
    xblock = [
        np.array([xstart[ix], xstart[ix] + block_size[1]])
        for _ in range(ny)
        for ix in range(nx)
    ]

    NRsm = kernelD2(xs=torch.arange(nx), ys=torch.arange(ny)).T.numpy()
    Kmat, nup = mat_upsample(lpad=lpad, subpixel=subpixel)

    return yblock, xblock, [ny, nx], block_size, NRsm, Kmat, nup


def compute_masks_ref_smooth_fft(refImg0, maskSlope, smooth_sigma,
                                 yblock, xblock):
    """
    Compute per-block taper masks, offsets, and FFT-smoothed reference images for
    nonrigid phase-correlation registration.
    This function extracts blocks from a full 2D reference image, applies a
    spatial taper (window) to each block, computes a per-block constant offset
    to compensate for masked/background regions, and computes a Gaussian-smoothed
    version of each block in the frequency domain (complex FFT) for use in
    phase-correlation based registration.

    PARAMETERS
        refImg0 : torch.Tensor
            2D reference image array of shape (Ly_full, Lx_full). Expected numeric
            image type (e.g. uint16, float32 or torch tensor). The function will
            extract sub-blocks using the indices supplied in yblock and xblock.

        maskSlope : float
            Scalar parameter controlling the slope of the sigmoid of the spatial taper. 
            Higher values increase tapered region size.

        smooth_sigma : float
            Standard deviation (in pixels) of the Gaussian smoothing applied to each
            block. Smoothing is performed in the frequency domain (via ref_smooth_fft). 
            Typical values are >= 0. A value of 0 should behave as no
            smoothing (identity).

        yblock : list[numpy.ndarray]
            List of length (ny * nx) giving the vertical (row) slice for each block.
            Each element is a 1D integer numpy array [y_start, y_end] specifying the
            inclusive start (y_start) and exclusive end (y_end) indices of the block
            along the vertical axis. Blocks are ordered row-major by block-grid row
            (iy) then column (ix): block_idx = iy * nx + ix.

        xblock : list[numpy.ndarray]
            List of length (ny * nx) giving the horizontal (column) slice for each
            block. Each element is a 1D integer numpy array [x_start, x_end]
            specifying the inclusive start and exclusive end indices along the
            horizontal axis. Ordering matches yblock (row-major block-grid order).

    OUTPUT
        maskMul_block : torch.Tensor
            Float32 tensor of shape (nb, Ly, Lx). Per-block multiplicative taper
            masks obtained by multiplying a local block taper.

        maskOffset_block : torch.Tensor
            Float32 tensor of shape (nb, Ly, Lx). Per-block additive offset fields
            computed as block_mean * (1 - maskMul_block) so that masked regions are
            filled with the local block mean scaled by the complement of the taper.
            
        cfRefImg_block : torch.Tensor (complex64)
            Complex32 tensor of shape (nb, Ly, Lx). Frequency-domain (FFT) representation
            of the Gaussian-smoothed reference blocks (output of ref_smooth_fft). These
            are intended for use in phase-correlation registration.
    
    """
    nb, Ly, Lx = len(yblock), yblock[0][1] - yblock[0][0], xblock[0][1] - xblock[0][0]
    dims = (nb, Ly, Lx)
    cfRef_dims = dims
    cfRefImg1 = torch.zeros(cfRef_dims, dtype=torch.complex64)

    maskMul = spatial_taper(maskSlope, *refImg0.shape)
    maskMul1 = torch.zeros(dims, dtype=torch.float)
    maskMul1[:] = spatial_taper(2 * smooth_sigma, Ly, Lx)
    maskOffset1 = torch.zeros(dims, dtype=torch.float)
    for yind, xind, maskMul1_n, maskOffset1_n, cfRefImg1_n in zip(
            yblock, xblock, maskMul1, maskOffset1, cfRefImg1):
        ix = np.ix_(
            np.arange(yind[0], yind[-1]).astype("int"),
            np.arange(xind[0], xind[-1]).astype("int"))
        refImg = refImg0[ix]

        # mask params
        maskMul1_n *= maskMul[yind[0] : yind[-1], xind[0] : xind[-1]]
        maskOffset1_n[:] = (refImg.float().mean() * (1. - maskMul1_n))

        # gaussian filter
        cfRefImg1_n[:] = ref_smooth_fft(refImg, smooth_sigma)
        
    return maskMul1, maskOffset1, cfRefImg1