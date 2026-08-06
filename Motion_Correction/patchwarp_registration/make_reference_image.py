import numpy as np
import cv2
import pandas as pd

from . import pyramid_registration


def make_reference_image(
    image_stack,
    threshold,
    fftdenoise,
    rigid_norm_method,
    rigid_norm_radius,
    rigid_template_center_frac,
):
    """
    Helper function to make the initial reference image

    PARAMETERS
        image_stack : np.ndarray
            Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.

        threshold : float
            Quantile threshold for selecting frames used for template image

        fftdenoise : bool
            Remove strong period noise (i.e., ripple PMT noise) from the template image

        rigid_norm_method : str
            Normalization method. Can be 'rank' or 'local'

        rigid_norm_radius : int
            Radius of normalization filter for the local method

        rigid_template_center_frac : float
            Central fraction of template used for registration

    OUTPUT
        template : np.ndarray
            Template image of shape (Ly, Lx)
    """

    # make the initial template
    initial_template = make_template(image_stack, threshold, fftdenoise)

    # Calculate register the shifts
    ymax, xmax, _ = pyramid_registration.compute_shifts(
        refImg=initial_template,
        f_reg=image_stack,
        rigid_norm_method=rigid_norm_method,
        rigid_norm_radius=rigid_norm_radius,
        rigid_template_center_frac=rigid_template_center_frac,
    )

    # Apply the shifts
    corrected_stack = pyramid_registration.shift_frames(
        f_reg=image_stack,
        yoff=ymax,
        xoff=xmax,
    )

    # Re-estimate the template with the corrected frames
    final_template = make_template(corrected_stack, threshold, fftdenoise)

    return final_template


def make_template(image_stack, threshold, fftdenoise):
    """Helper function to make the initial template from multiple frames

    PARAMETERS
        image_stack : np.ndarray
            Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.

        threshold : float
            Quantile threshold for selecting frames used for template image

        fftdenoise : bool
            Remove strong period noise (i.e., ripple PMT noise) from the template image

    OUTPUT
        template : np.ndarray
            Template image of shape (Ly, Lx)

    """
    reduction_ratio = np.array([16, 16])
    max_iter = 5

    nframes, Ly, Lx = image_stack.shape

    sLy = np.ceil(Ly / reduction_ratio[0])
    sLx = np.ceil(Lx / reduction_ratio[1])

    # reduce image size for correlation
    small_stack = np.zeros((nframes, sLy, sLx))
    for i in range(nframes):
        small_stack[i, :, :] = cv2.resize(
            image_stack[i, :, :], (sLy, sLx), interpolation=cv2.INTER_CUBIC
        )

    # Identify selected frames
    image_2d = np.reshape(small_stack, (nframes, sLy * sLx))
    selected = np.ones((nframes, 1), dtype=bool)

    # Repeatedly update reference frames based on their correlation
    for iter in range(max_iter):
        # Get average to compare against
        mean_2d = np.nanmean(image_2d[selected, :], axis=0)
        # Correlate agains the average
        c = np.zeros((nframes, 1))
        for i in range(nframes):
            c[i] = np.corrcoef(image_2d[i, :], mean_2d)[0, 1]

        # Update the selected frames
        selected = c > np.quantile(c, 1 - threshold)

    # Define the reference iamge
    template = np.nanmean(image_stack[selected, :, :], axis=0)

    # Denoise if specified
    if fftdenoise:
        freqmap_mean = np.fft.fftshift(np.fft.fft2(template))
        freqmap_mean_amp = np.log(np.abs(freqmap_mean))

        dn_threshold = 0.5

        # Zero the midline roes
        mid = matlab_round(Ly / 2)
        freqmap_mean_amp_midlines0 = freqmap_mean_amp.copy()
        freqmap_mean_amp_midlines0[mid - 3 : mid + 2, :] = 0

        # Column-wise max
        power_max1 = freqmap_mean_amp_midlines0.max(axis=0)

        # Calcualte median filter
        f_y = (
            pd.Series(power_max1)
            .rolling(window=10, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        power_max1_detrended = power_max1 - f_y
        locs_max1 = np.nonzero(power_max1_detrended > dn_threshold)[0]
        # Repeat for other dimension
        mid = matlab_round(Lx / 2)
        freqmap_mean_amp_midlines0 = freqmap_mean_amp.copy()
        freqmap_mean_amp_midlines0[:, mid - 3 : mid + 2] = 0

        # Column-wise max
        power_max2 = freqmap_mean_amp_midlines0.max(axis=1)

        # Calcualte median filter
        f_y = (
            pd.Series(power_max2)
            .rolling(window=10, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        power_max2_detrended = power_max2 - f_y
        locs_max2 = np.nonzero(power_max2_detrended > dn_threshold)[0]

        remove_mask = np.zeros((freqmap_mean_amp.size))
        remove_range = 3
        if len(locs_max1) > 0:
            for i in range(len(locs_max1)):
                if (locs_max1[i] - remove_range) < 1:
                    remove_mask[:, 0 : locs_max1[i] + remove_range] = 1
                elif (locs_max1 + remove_range) > Lx:
                    remove_mask[:, locs_max1[i] - remove_range : Lx] = 1
                else:
                    remove_mask[
                        : locs_max1[i] - remove_range : locs_max1[i] + remove_range
                    ] = 1

        if len(locs_max2) > 0:
            for i in range(len(locs_max2)):
                if (locs_max2[i] - remove_range) < 1:
                    remove_mask[0 : locs_max2[i] + remove_range, :] = 1
                elif (locs_max2 + remove_range) > Ly:
                    remove_mask[locs_max2[i] - remove_range : Ly, :] = 1
                else:
                    remove_mask[
                        locs_max2[i] - remove_range : locs_max2[i] + remove_range, :
                    ] = 1

        x = 2
        remove_mask[
            matlab_round(Ly / 2) - x : matlab_round(Ly / 2) + x,
            matlab_round(Lx / 2) - x : matlab_round(Lx / 2) + x,
        ] = 0

        remove_mask = remove_mask.astype(bool)

        temp = np.fft.fftshift(np.fft.fft2(template))
        temp[remove_mask] = 0

        template = np.abs(np.fft.ifft2(np.fft.ifftshift(temp)))

    return template


def matlab_round(x):
    """Function to replicate matlab rounding behavior"""
    return int(np.floor(x + 0.5)) if x >= 0 else int(np.ceil(x - 0.5))
