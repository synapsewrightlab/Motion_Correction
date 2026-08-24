import numpy as np
import warnings

from scipy.signal import convolve2d
from scipy.ndimage import shift as subshift

from ..io import binary


def pyramid_parallel(args):
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
        pyramid_args,
    ) = args

    rigid_norm_method, rigid_norm_radius, rigid_template_center_frac = pyramid_args

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
    yoff, xoff, corrXY = compute_shifts(
        reference_image,
        batch_frames,
        rigid_norm_method,
        rigid_norm_radius,
        rigid_template_center_frac,
    )

    # Apply the shifts
    corrected = shift_frames(batch_frames, yoff=yoff, xoff=xoff)
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
        corrected_alt = shift_frames(alt_frames, yoff=yoff, xoff=xoff)
        meanImg_alt = np.nanmean(corrected_alt, axis=0)
        f_out_alt[start:end] = corrected_alt
        f_out_alt.file.flush()

        f_in_alt.close()
        f_out_alt.close()

    return yoff, xoff, corrXY, meanImg, meanImg_alt


def compute_shifts(
    refImg,
    f_reg,
    rigid_norm_method,
    rigid_norm_radius,
    rigid_template_center_frac,
):
    """
    Function to compute the shifts using a pyrammid-based hill-climbing approach

    PARAMETERS
        refImg : np.ndarray
            Reference image for alignment, of shape (Ly, Lx)

        f_reg : np.ndarray
            Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.

        rigid_norm_method : str
            Normalization method. Can be 'rank' or 'local'

        rigid_norm_radius : int
            Radius of normalization filter for the local method

        rigid_template_center_frac : float
            Central fraction of template used for registration

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

    if rigid_norm_method == "rank":
        refImg = rank_transform(refImg)
        f_reg = rank_transform(f_reg)

    elif rigid_norm_method == "local":
        refImg = imnormalize(refImg, rigid_norm_radius)
        f_reg = imnormalize(f_reg, rigid_norm_radius)

    # Use on the center of the image for calculating shifts
    center_target = get_center(refImg, rigid_template_center_frac)

    max_iter = np.ceil(np.sum(refImg.size - center_target.size) / 2)

    for i in range(nframes):
        # Get center of the source frame
        center_source = get_center(f_reg[i, :, :], rigid_template_center_frac)

        # Build the pyramid
        target_pyramid = build_pyramid(center_target, n_levels=3)
        source_pyramid = build_pyramid(center_source)

        # Perform pyramid registration step
        shift, score = pyramid_step(target_pyramid, source_pyramid, max_iter=max_iter)

        # Perform bilinear interpolation of the shifts
        shift, score = bilinear_step(
            center_target, center_source, dy=shift[0], dx=shift[1]
        )

        ymax[i] = shift[0]
        xmax[i] = shift[1]
        cmax[i] = score

    return ymax, xmax, cmax


def shift_frames(f_reg, yoff, xoff):
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


def bilinear_step(target, source, dy, dx):
    """Helper function to perform the bilinear interpolation shift step"""
    cc = np.zeros((3, 3))

    cc[0, 1] = covariance_score(target, source, dy=dy - 1, dx=dx)
    cc[1, 0] = covariance_score(target, source, dy=dy, dx=dx - 1)
    cc[1, 1] = covariance_score(target, source, dy=dy, dx=dx)
    cc[1, 2] = covariance_score(target, source, dy=dy, dx=dx + 1)
    cc[2, 1] = covariance_score(target, source, dy=dy + 1, dx=dx)

    d, c = sub_pixel_interpolate(cc)

    final_shift = (dy + d[0], dx + d[1])

    return final_shift, c


def pyramid_step(target_pyramid, source_pyramid, max_iter):
    """Helper function to perform the pyramid registration step"""
    shift = (0, 0)
    # Get source pyramid
    for level_target, level_source in zip(target_pyramid, source_pyramid):
        shift, score = hill_climb(
            level_target, level_source, shift, max_step=8, min_step=1, max_iter=max_iter
        )
        shift = (shift[0] * 2, shift[1] * 2)

    final_shift = (shift[0] // 2, shift[1] // 2)

    return final_shift, score


def sub_pixel_interpolate(arr):
    """Helper function to estimate quadratic interpolation for subpixel shift estimation"""
    assert arr.shape == (3, 3), "Wrong matrix size. Input should be numerical 3x3."

    top = arr[0, 1]
    left = arr[1, 0]
    center = arr[1, 1]
    right = arr[1, 2]
    bottom = arr[2, 1]

    if np.any(np.isnan([top, left, center, right, bottom])):
        warnings.warn("Cannot handle NaN values")
        return np.array([0.0, 0.0]), center

    assert (
        np.max([left, top, center, bottom, right]) == center
    ), "Wrong matrix. Should be centered on max"

    v = np.array([top, left, center, right, bottom])

    coeff = np.array(
        [
            [1 / 4, 1 / 4, -1, 1 / 4, 1 / 4],
            [
                -1 / 2,
                0,
                0,
                0,
                1 / 2,
            ],
            [0, -1 / 2, 0, 1 / 2, 0],
            [0, 0, 1, 0, 0],
        ]
    )

    p1, p2, p3, p4 = coeff @ v

    if p1 >= 0:
        return np.array([0.0, 0.0]), center

    d = np.array([-p2 / (2 * p1), -p3 / (2 * p1)])
    c = p4 - (p2**2 + p3**2) / (4 * p1)

    return d, c


def hill_climb(target, source, initial_shift, max_step=8, min_step=1, max_iter=10):
    """Helper function to perform hill-climbing pattern search"""
    dy, dx = initial_shift
    step = max_step
    best_score = covariance_score(target, source, dy, dx)

    for _ in range(max_iter):
        candidates = [
            (dy + s * step, dx + t * step)
            for s in (-1, 0, 1)
            for t in (-1, 0, 1)
            if not (s == 0 and t == 0)
        ]
        scores = [
            (covariance_score(target, source, cy, cx), (cy, cx))
            for cy, cx in candidates
        ]
        best_candidate_score, best_candidate = max(scores, key=lambda p: p[0])

        if best_candidate_score > best_score:
            best_score = best_candidate_score
            dy, dx = best_candidate

        else:
            step //= 2
            if step < min_step:
                break

    return (dy, dx), best_score


def covariance_score(target, source, dy, dx):
    """Helper function to calculate the covariance between two frames"""
    shifted = np.roll(source, shift=(dy, dx), axis=(0, 1))

    # Crop to a valid overlap so wrap around pixels don't pollute the image
    h, w = target.shape
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)

    t = target[y0:y1, x0:x1]
    s = shifted[y0:y1, x0:x1]

    t_c = t - t.mean()
    s_c = s - s.mean()

    cov = np.sum(t_c * s_c) / t_c.size

    return cov


def build_pyramid(frame, n_levels=3):
    """Helper function to build the pyramid of images for registration"""
    pyramid_multiply = 2

    pyramid = [frame]
    current = frame
    for _ in range(n_levels):
        current = box_downsample(current, pyramid_multiply)
        pyramid.append(current)

    return pyramid[::-1]


def box_downsample(img, factor):
    """Helper function to downsample images for the pyramid"""
    h, w = img.shape
    h2 = h - h % factor
    w2 = w - w % factor
    img = img[:h2, :w2]
    return img.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def get_center(img, factor):
    """Helper function to get the center of the image for registration"""
    h, w = img.shape
    new_h = int(round(h * factor))
    new_w = int(round(w * factor))

    start_h = (h - new_h) // 2
    start_w = (w - new_w) // 2

    return img[start_h : start_h + new_h, start_w : start_w + new_w]


def rank_transform(img_stack):
    """Function to perform rank normalization of images"""

    # deal with 2d arrays
    squeeze_back = img.ndim == 2
    if squeeze_back:
        img = img[np.newaxis, :, :]

    result = np.zeros(img_stack.size)
    for i in range(img_stack.size[0]):
        img = img_stack[i, :, :]
        _, _, ic = np.unique(img, return_index=True, return_inverse=True)
        result[i, :, :] = ic.reshape(img.shape)

    if squeeze_back:
        result = result[0, :, :]

    return result


def imnormalize(img, radius):
    """Helper function to perform local normalization of images"""

    # deal with 2d arrays
    squeeze_back = img.ndim == 2
    if squeeze_back:
        img = img[np.newaxis, :, :]

    f = matlab_fspecial_disk(radius)

    n_cells = convolve2d(np.ones(img.shape[1:]), f, mode="same")

    imsm = np.zeros(img.shape, dtype=np.float64)
    for i in range(img.shape[0]):
        imsm[i, :, :] = (
            convolve2d(img[i, :, :].astype(np.float64), f, mode="same") / n_cells
        )

    imn = img.mean() * img / imsm

    if squeeze_back:
        imn = imn[0, :, :]

    return imn


def matlab_fspecial_disk(radius):
    """Function to approximate matlabs fspecial('disk') function"""
    crad = int(np.ceil(radius - 0.5))

    x, y = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.maximum(np.abs(x), np.abs(y)).astype(float)
    minxy = np.minimum(np.abs(x), np.abs(y)).astype(float)

    m1 = np.where(
        radius**2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2,
        minxy - 0.5,
        np.sqrt(np.maximum(radius**2 - (maxxy + 0.5) ** 2, 0)),
    )

    m2 = np.where(
        radius**2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2,
        minxy + 0.5,
        np.sqrt(np.maximum(radius**2 - (maxxy + 0.5) ** 2, 0)),
    )

    a1 = np.arcsin(np.clip(m1 / radius, -1, 1))
    a2 = np.arcsin(np.clip(m2 / radius, -1, 1))
    sgrid = (
        radius**2 * (0.5 * (a2 - a1) + 0.25 * (np.sin(2 * a2) - np.sin(2 * a1)))
        - (maxxy - 0.5) * (m2 - m1)
        + (m1 + minxy + 0.5)
    )

    cond = (
        (radius**2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2)
        & (radius**2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)
    ) | ((minxy == 0) & (maxxy - 0.5 < radius) & (maxxy + 0.5 >= radius))

    sgrid = sgrid * cond
    sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < radius**2)

    c = crad
    sgrid[c, c] = min(np.pi * radius**2, np.pi / 2)

    # Edge conditions
    if crad > 0 and radius > crad - 0.5 and radius**2 < (crad - 0.5) ** 2 + 5:
        m1v = np.sqrt(radius**2 - (crad - 0.5) ** 2)
        m1n = m1v / radius
        sg0 = 2 * (
            radius**2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n)))
            - m1v * (crad - 0.5)
        )
        sgrid[2 * crad, c] = sg0
        sgrid[c, 2 * crad] = sg0
        sgrid[c, 0] = sg0
        sgrid[0, c] = sg0
        sgrid[2 * crad - 1, c] -= sg0
        sgrid[c, 2 * crad - 1] -= sg0
        sgrid[c, 1] -= sg0
        sgrid[1, c] -= sg0

    sgrid[c, c] = min(sgrid[c, c], 1)

    sgrid = sgrid / sgrid.sum()

    return sgrid
