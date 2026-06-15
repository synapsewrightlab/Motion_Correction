import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
import torch

try:
    # pytorch > 1.7
    from torch.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift
except:
    # pytorch <= 1.7
    raise ImportError("pytorch version > 1.7 required")

from . import cellpose_transforms as transforms

eps = torch.complex(torch.tensor(1e-5), torch.tensor(0.0))


def convolve(mov, img):
    """
    Convolve a 3d frame sequence by a 2d image in the Fourier domain using
    phase correlations
    
    Applies FFT to each frame, normalizes by magnitude, multiplies by im, and returns
    inverse FFT

    PARAMETERS
        mov: torch.Tensor
            Input frames of shape (nImg, Ly, Lx)

        img: torch.Tensor
            2d complex-values convolution kernels of shape (Ly, Lx)
    
    OUTPUT
        convolved_data: torch.Tensor
            Real-valued convolution result (nImg, Ly, Lx)
    """
    mov = fft2(mov)
    mov /= (eps + torch.abs(mov))
    mov *= img
    mov = torch.real(ifft2(mov))

    return mov

def spatial_taper(sig, Ly, Lx):
    """
    Compute spatial taper mask using a sigmoid function on the image edges

    Creates a 2D multiplicative mask that smoothly reduces values near the
    edges. Controlled by Gaussian-like sigmoid with standard deviation'sig'.

    PARAMETERS
        sig: float
            Scalar parameters controlling the slope of the sigmoid taper.
            Higher values increas the size of the tapered boarder region

        Ly: int
            Frame height in pixels

        Lx: int
            Frame width in pixels

    OUTPUT
        maskMul: torch.Tensor
            Floating-point multiplicative mas of shape (Ly, Lx), with values 
            near 1.0 in the center and smoothly decaying to 0.0 at edges
    
    """
    y = torch.arange(0, Ly, dtype=torch.double)
    x = torch.arange(0, Lx, dtype=torch.double)
    x = (x - x.mean()).abs()
    y = (y - y.mean()).abs()
    mY = ((Ly - 1) / 2) - 2 * sig
    mX = ((Lx - 1) / 2) - 2 * sig
    maskY = 1.0 / (1.0 + torch.exp((y - mY) / sig))
    maskX = 1.0 / (1.0 + torch.exp((x - mX) / sig))

    maskMul = maskY[:, None] * maskX

    return maskMul

def temporal_smooth(data, sigma):
    """
    Apply 1d Gaussian smoothing along time axis of 3d array

    PARAMETERS
        data: np.ndarray
            Input data of shape (nimg, Ly, Lx) to be smoothed

        sigma: float
            Std of the Gaussian kernel used for smoothing

    OUTPUT
        smoothed_data: np.ndarray
            Temporally smoothed data of shape (nimg, Ly, Lx)
    
    """
    smoothed_data = gaussian_filter1d(data, sigma=sigma, axis=0)

    return smoothed_data

def complex_fft2(img):
    """
    Compute the complex conjugate of the 2d FFT of an image

    PARAMETERS
        img: torch.Tensor
            2d input image of shape (Ly, Lx)
    
    OUTPUT
        cfImg: torch.Tensor
            Complex conjugate of the 2d FFT, shape (Ly, Lx)
    """
    cfImg = torch.conj(fft2(img))

    return cfImg

def gaussian_kernel(sigma_y, sigma_x, Ly, Lx, device=torch.device("cpu")):
    """
    Generate a normalized 2d gaussian kernel

    PARAMETERS
        sigma_y: float
            Std of Gaussian along y axis
        
        sigma_x: float
            Std of Gaussian along the x axis

        Ly: int
            Number of pixels in y axis

        Lx: int
            Number of pixels in the x axis

        device: torch.device, default cpu

    OUTPUT
        kernel: torch.Tensor
            Normalized 2d Gaussian keranl of shape (Lx, Ly), summing to 1.0
    
    """
    y = torch.arange(0, Ly, device=device, dtype=torch.float)
    y -= y.mean()
    x = torch.arange(0, Lx, device=device, dtype=torch.float)
    x -= x.mean()

    ky = torch.exp(-y**2 / (2 * sigma_y**2))
    kx = torch.exp(-x**2 / (2 * sigma_x**2))

    kernel = ky[:, None] * kx
    kernel /= kernel.sum()

    return kernel

def gaussian_fft(sig, Ly, Lx):
    """
    Compute the real-valued FFT of a 2d isotropic Gaussian kernel for smoothing

    PARAMETERS
        sig: float
            Std (in pixels) of the isotropic Gaussian kernel

        Ly: int
            Number of pixels in y axis

        Lx: int
            Number of pixels in x axis

    OUTPUT
        gaussian_fft: torch.Tensor
            Real-valued 2d FFT of Gaussian kernel, shape (Ly, Lx)
    
    """
    kernel = gaussian_kernel(sig, sig, Ly, Lx)
    g_fft = torch.real(fft2(ifftshift(kernel)))

    return g_fft

def ref_smooth_fft(refImg, smooth_sigma=None):
    """
    Compute the smoothed, normalized complex-conjugated FFT of a reference image
    for phase-correlation

    Takes 2d FFT complex conjugate of 'refImg', whitens, and multiplies by 
    Gaussian filter in the frequency domain with std 'smooth_sigma'

    PARAMETERS
        refImg: torch.Tensor
            2d referenc iamge of shape (Ly, Lx)
        
        smooth_sigma: float, optional
            Std in pixels of Gaussain smootiong applied. If None, no smoothing
            is done

    OUTPUT
        cfRefImg: torch.Tensor
            Complex64 tensor of shape (Ly, Lx) containing the smoothed, whitened
            complex-conjugated FFT of the reference image
    
    """
    cfRefImg = complex_fft2(img=refImg)
    cfRefImg /= (1e-5 + torch.abs(cfRefImg))
    if smooth_sigma is not None:
        cfRefImg *= gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
    
    return cfRefImg.type(torch.complex64)

def kernelD(xs, ys, sigL=0.85):
    """
    Compute Gaussian interpolation kernel between two sets of 2d grid coordinates

    Builds a kernel matrix K where K[i, j] = exp(-d^2 / (2 * sigL^2)) with d being the
    Euclidean distance between the i-th point in `xs` x `xs` and the j-th point in
    `ys` x `ys`. Used for sub-pixel up-sampling in registration.

    PARAMETERS
        xs: torch.Tensor
            1d tensor of grid coordinates of the source points
        
        ys: torch.Tensor
            1d tensor of grid coordinates of the target points

        sigL: float, default 0.85
            Smoothing width of Gaussian kernel

    OUTPUT
        K: torch.Tensor
            Gaussian kernel matrix of shape (len(xs)**2, len(ys)**2)

    """
    xs0, xs1 = torch.meshgrid(xs, xs, indexing="ij")
    ys0, ys1 = torch.meshgrid(ys, ys, indexing="ij")
    dxs = xs0.reshape(-1, 1) - ys0.reshape(1, -1)
    dys = xs1.reshape(-1, 1) - ys1.reshape(1, -1)
    K = torch.exp(-(dxs**2 + dys**2) / (2 * sigL**2))

    return K


def kernelD2(xs, ys):
    """
    Compute a normalized Gaussian kernel matrix from two 1D coordinate tensors.

    Builds a 2D meshgrid from `xs` and `ys`, computes pairwise Gaussian distances
    between all flattened grid points, and row-normalizes the result. It is used 
    for smoothing phase-correlation maps across blocks.

    PARAMETERS
        xs : torch.Tensor
            1D tensor of grid coordinates along one axis.

        ys : torch.Tensor
            1D tensor of grid coordinates along the other axis.

    OUTPUT
        R : torch.Tensor
            Row-normalized Gaussian kernel matrix of shape (N, N) where N = len(xs) * len(ys).
    """
    ys, xs = torch.meshgrid(xs, ys, indexing="ij")
    ys = ys.flatten().reshape(1, -1)
    xs = xs.flatten().reshape(1, -1)
    R = torch.exp(-((ys - ys.T)**2 + (xs - xs.T)**2))
    R = R / torch.sum(R, axis=0)

    return R


def mat_upsample(lpad, subpixel=10, device=torch.device("cpu")):
    """
    Build an interpolation matrix for sub-pixel upsampling of correlation peaks.

    Constructs a Gaussian interpolation matrix (`Kmat`) that maps from the original
    integer grid of size (2*lpad+1) to a finer grid with spacing 1/subpixel, by solving
    a linear system using `kernelD`.

    PARAMETERS
        lpad : int
            Half-width of the integer grid. The grid spans from -lpad to +lpad.

        subpixel : int, optional (default 10)
            Up-sampling factor. The output grid has spacing 1/subpixel.

        device : torch.device, optional (default torch.device("cpu"))
            Device on which to create the grid tensors.

    OUTPUT  
        Kmat : torch.Tensor
            Interpolation matrix of shape ((2*lpad+1)**2, nup**2) mapping the original grid
            to the up-sampled grid.
        nup : int
            Number of points along one axis of the up-sampled grid.
    """
    xs = torch.arange(-lpad, lpad + 1, device=device)
    xs_up = torch.arange(-lpad, lpad + .001, 1. / subpixel, device=device)
    kernel0 = kernelD(xs, xs)
    kernel_up = kernelD(xs, xs_up) 
    Kmat = torch.linalg.solve(kernel0, kernel_up)
    nup = len(xs_up)

    return Kmat, nup


def highpass_mean_image(I, aspect=1.):
    """
    Compute an enhanced mean image by applying a high-pass Gaussian filter.

    Subtracts low-frequency content using a Gaussian kernel (sigma=3 in each axis,
    scaled by `aspect` in y), then rescales the result to [0, 1] using the 1st and
    99th percentiles.

    PARAMETERS
        I : numpy.ndarray
            2D mean image of shape (Ly, Lx).

        aspect : float, optional (default 1.0)
            Aspect ratio correction factor. Values != 1.0 scale the Gaussian sigma along
            the y-axis by this factor.

    OUTPUT
        img_filt : numpy.ndarray
            High-pass filtered image of shape (Ly, Lx), clipped to [0, 1].
    """
    Ly, Lx = I.shape
    img_filt = cv2.resize(I, (Lx, int(np.round(Ly * aspect))))
    img_filt = transforms.normalize_img(img_filt[..., np.newaxis], sharpen_radius=3)
    img_filt = cv2.resize(img_filt.squeeze(), (Lx, Ly))
    img_filt = np.clip(img_filt, 0, 1)

    return img_filt

