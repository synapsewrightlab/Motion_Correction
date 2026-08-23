import numpy as np
import cv2
import pandas as pd

from . import cv_registration


def make_reference_image(
    image_stack,
    iterations,
    corr_thresh,
    factor,
    marginh,
    marginw,
    sigma_smoothing,
    sigma_normalization,
    normalization_offset,
    to_equalize_histogram,
):
    """
    Helper function to make the initial reference image

    PARAMETERS
        image_stack : np.ndarray
            Input frames of shape (n_frames, Ly, Lx), supporting index slicing

        corr_thresh: float
            Quantile threshold for selecting frames used for template image

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

    OUTPUT
        template : np.ndarray
            Template image of shape (Ly, Lx)

    """
    # Make the initial tamplate
    template = make_template(image_stack, corr_thresh, factor)

    # Register
    for _ in range(iterations):
        ymax, xmax, _ = cv_registration.cv_compute_shifts(
            refImg=template,
            f_reg=image_stack,
            factor=factor,
            marginh=marginh,
            marginw=marginw,
            sigma_smoothing=sigma_smoothing,
            sigma_normalization=sigma_normalization,
            normalization_offset=normalization_offset,
            to_equalize_histogram=to_equalize_histogram,
        )

        corrected = cv_registration.cv_shift_frames(
            f_reg=image_stack,
            yoff=ymax,
            xoff=xmax,
        )

        template = make_template(corrected, corr_thresh, factor)

    return template


def make_template(image_stack, corr_thresh, factor):
    """Helper function to make initial template"""
    reduction_ratio = factor * 2
    max_iter = 5

    nframes, Ly, Lx = image_stack.shape

    sLy = int(np.ceil(Ly / reduction_ratio))
    sLx = int(np.ceil(Lx / reduction_ratio))

    # reduce image size for correlation
    small_stack = np.zeros((nframes, sLy, sLx))
    for i in range(nframes):
        small_stack[i, :, :] = cv2.resize(
            image_stack[i, :, :],
            (sLx, sLy),
            interpolation=cv2.INTER_AREA,
        )

    # Identify correlated frames
    image_2d = np.reshape(small_stack, (nframes, sLy * sLx))
    selected = np.ones(nframes, dtype=bool)

    # Iteratively search for correlated frames
    for _ in range(max_iter):
        # Get average to compare against
        mean_2d = np.nanmean(image_2d[selected, :], axis=0)
        # Correlate against the average
        c = np.zeros(nframes)
        for i in range(nframes):
            c[i] = np.corrcoef(image_2d[i, :], mean_2d)[0, 1]

        # Update the selected frames
        selected = c > np.quantile(c, 1 - corr_thresh)

    # Define the reference image
    template = np.nanmean(image_stack[selected, :, :], axis=0)
