import numpy as np
import warnings

import cv2

from scipy.signal import convolve2d
from scipy.ndimage import shift as subshift

from ..io import binary

from .subpix import min_max_loc_subpix


def cv_parallel(args):
    """
    Function to perform pyramid registration across parallelized chunks

    PARAMETERS
        args : tuple
            Tuple containing the following variables

            input_path : str
                File path to the f_align_in file

            output_path : str
                File path to the f_align_out file

            input_path_alt : str
                Optional: File path to the f_alt_in file

            output_path_alt : str
                Optional: File path to the f_alt_out file

            input_shape : tuple
                File size (nframes, Ly, Lx)

            output_shape : tuple
                File size (nframes, Ly, Lx)

            input_shape_alt : tuple
                Optional: File size (nframes, Ly, Lx)

            output_shape_alt : tuple
                Optional: File size (nframes, Ly, Lx)

            start : int
                Starting frame idx of the chunk to be processed

            end : int
                Ending frame idx of the chunk to be processed

            reference_image : np.ndarray
                2D array (Ly, Lx) of the reference image

            cv_args : tuple
                Contains values to align frames

    OUTPUTS
         ymax : np.ndarray
            1-D integer array of length N with the y (row) shift for each frame that maximizes the
            phase-correlation.

        xmax : np.ndarray
            1-D integer array of length N with the x (column) shift for each frame that maximizes the
            phase-correlation.

        cmax : np.ndarray
            1-D array of length N containing the maximum phase-correlation value found for each frame.

    """
    # Upack the arguments
    (
        input_path,
        output_path,
        input_path_alt,
        output_path_alt,
        input_shape,
        output_shape,
        input_shape_alt,
        output_shape_alt,
        start,
        end,
        reference_image,
        cv_args,
    ) = args

    (
        factor,
        marginh,
        marginw,
        sigma_smoothing,
        sigma_normalization,
        normalization_offset,
        to_equalize_histogram,
    ) = cv_args

    # Create memmaps to relevant file
    f_in = binary.BinaryFile(
        Ly=input_shape[1],
        Lx=input_shape[2],
        filename=input_path,
        n_frames=input_shape[0],
        write=False,
    )
    f_out = binary.BinaryFile(
        Ly=output_shape[1],
        Lx=output_shape[2],
        filename=output_path,
        n_frames=output_shape[0],
        write=True,
    )

    batch_frames = f_in[start:end].copy()

    # Compute the shifts
    yoff, xoff, corrXY = cv_compute_shifts(
        reference_image,
        batch_frames,
        factor,
        marginh,
        marginw,
        sigma_smoothing,
        sigma_normalization,
        normalization_offset,
        to_equalize_histogram,
    )

    # Apply the shifts
    corrected = cv_shift_frames(batch_frames, yoff=yoff, xoff=xoff)
    meanImg = np.nanmean(corrected, axis=0)
    meanImg_alt = None

    # Write to disk
    f_out[start:end] = corrected
    f_out.file.flush()

    f_in.close()
    f_out.close()

    if input_path_alt:
        f_in_alt = binary.BinaryFile(
            Ly=input_shape_alt[1],
            Lx=input_shape_alt[2],
            filename=input_path_alt,
            n_frames=input_shape_alt[0],
            write=False,
        )
        f_out_alt = binary.BinaryFile(
            Ly=output_shape_alt[1],
            Lx=output_shape_alt[2],
            filename=output_path_alt,
            n_frames=output_shape_alt[0],
            write=True,
        )

        alt_frames = f_in_alt[start:end].copy()

        # Shift frames
        corrected_alt = cv_shift_frames(alt_frames, yoff=yoff, xoff=xoff)
        meanImg_alt = np.nanmean(corrected_alt, axis=0)
        f_out_alt[start:end] = corrected_alt
        f_out_alt.file.flush()

        f_in_alt.close()
        f_out_alt.close()

    return yoff, xoff, corrXY, meanImg, meanImg_alt


def cv_compute_shifts(
    refImg,
    f_reg,
    factor=2,
    marginh=128,
    marginw=128,
    sigma_smoothing=0,
    sigma_normalization=0,
    normalization_offset=0,
    to_equalize_histogram=False,
):
    """
     Function to compute the shifts using a OpenCv architecture

    PARAMETERS
        refImg : np.ndarray
            Reference image for alignment, of shape (Ly, Lx)

        f_reg : np.ndarray
            Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.

        factor : int
            Scaling factor

        marginh : int
            Margin around the center of the image to use for alignment in y

        marginw : int
            Margin around the center of the image to use for alignment in x

        sigma_smoothing : float
            Value to use for sigma smoothing. set to 0 for no smoothing

        sigma_normalization : float
            Value to use for normalization. set to 0 for no normalization

        normalization_offset : float
            Value to offset the normalzation

        to_equalize_histogram : bool
            Whether to equalize the output histogram

    OUTPUTS
         ymax : np.ndarray
            1-D integer array of length N with the y (row) shift for each frame that maximizes the
            phase-correlation.

        xmax : np.ndarray
            1-D integer array of length N with the x (column) shift for each frame that maximizes the
            phase-correlation.

        cmax : np.ndarray
            1-D array of length N containing the maximum phase-correlation value found for each frame.

    """
    nframes = f_reg.shape[0]
    ymax = np.zeros(nframes)
    xmax = np.zeros(nframes)
    cmax = np.zeros(nframes)

    # Preprocess the template image
    template = refImg.copy()
    template = preprocess_img(
        template,
        factor,
        sigma_smoothing,
        sigma_normalization,
        normalization_offset,
        to_equalize_histogram,
    )

    # Get the center part of the image
    row_lo = int(round(marginh / factor))
    row_hi = int(round(refImg.shape[0] / factor - marginh / factor))
    col_lo = int(round(marginw / factor))
    col_hi = int(round(refImg.shape[1] / factor - marginw / factor))

    template_center = template[row_lo:row_hi, col_lo:col_hi].copy()

    for i in range(nframes):
        curr_img = f_reg[i, :, :]
        dy, dx, corr = align_frame(
            curr_img,
            template_center,
            marginh,
            marginw,
            factor,
            sigma_smoothing,
            sigma_normalization,
            normalization_offset,
            to_equalize_histogram,
        )

        ymax[i] = dy
        xmax[i] = dx
        cmax[i] = corr

    return ymax, xmax, cmax


def cv_shift_frames(f_reg, yoff, xoff):
    """
    Function to shift frames

    PARAMETERS
        f_reg : np.ndarray
            Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.

        yoff : np.ndarray
            1D array of the offsets in the y (row) dimension

        xoff : np.ndarray
            1D array of the offsets in the x (column) dimension

    OUTPUTS
        f_correct : np.ndarray
            Output frames of shape (n_frames, Ly, Lx) after shifts

    """
    # deal with 2d arrays
    squeeze_back = f_reg.ndim == 2
    if squeeze_back:
        f_reg = f_reg[np.newaxis, :, :]

    f_correct = np.zeros(f_reg.shape).astype(np.float64)

    for i in range(f_reg.shape[0]):
        f = f_reg[i, :, :]
        shifted = subshift(f, shift=(yoff[i], xoff[i]), order=3)
        f_correct[i, :, :] = shifted

    if squeeze_back:
        f_correct = f_correct[0, :, :]

    return f_correct


def preprocess_img(
    refImg,
    factor,
    sigma_smoothing,
    sigma_normalization,
    normalization_offset,
    to_equalize_histogram,
):
    """Helper function to first normalize the template image"""
    # Apply offset to reference iamge
    out = refImg.astype(np.float32) + normalization_offset

    # Resize the image
    inv_factor = 1.0 / factor
    out = cv2.resize(
        out, (0, 0), fx=inv_factor, fy=inv_factor, interpolation=cv2.INTER_AREA
    )

    if sigma_smoothing:
        tmp1 = cv2.GaussianBlur(out, (0, 0), sigma_smoothing / factor)
    else:
        tmp1 = out

    if sigma_normalization:
        tmp2 = cv2.GaussianBlur(out, (0, 0), sigma_normalization / factor)
        tmp2 = (tmp2 + normalization_offset).astype(np.float32)
        tmp2 = cv2.normalize(tmp2, None, alpha=1, norm_type=cv2.NORM_L1)
        out = cv2.divide(tmp1, tmp2)
    else:
        out = tmp1

    if to_equalize_histogram:
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
        out = out.astype(np.uint8)
        out = cv2.equalizeHist(out)

    return out


def align_frame(
    img,
    template,
    marginh,
    marginw,
    factor,
    sigma_smoothing,
    sigma_normalization,
    normalization_offset,
    to_equalize_histogram,
):
    """Helper fuction to compute the shifts for the alignment"""

    f_reg = preprocess_img(
        img,
        factor,
        sigma_smoothing,
        sigma_normalization,
        normalization_offset,
        to_equalize_histogram,
    )

    # Match the template
    result = cv2.matchTemplate(f_reg, template, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)  # max_loc = (x, y)
    subpix_loc, err = min_max_loc_subpix(result.astype(np.float32), max_loc)

    dx = subpix_loc[0] * factor - marginw
    dy = subpix_loc[1] * factor - marginh

    return -dy, -dx, max_val
