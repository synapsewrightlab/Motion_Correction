import math
import numpy as np
import cv2

from ..suite2p_registration import cellpose_transforms as transforms


def get_block_ids(arr_len, block_num):
    """Helper function to get the indexes to evenly divide the frames into blocks"""
    base_size = arr_len // block_num
    remainder = arr_len % block_num

    idxs = []
    start = 0

    for i in range(block_num):
        end = start + base_size + (1 if i < remainder else 0)
        idxs.append((start, end))
        start = end

    return idxs


def get_batch_ids(start, end, n_workers, max_step):
    """Helper function to set the number of frames in each batched based on the number
    of cpu workers
    """
    length = end - start

    base_size = math.ceil(length / n_workers)

    batch_size = min(base_size, max_step)

    batch_idxs = [(i, min(i + batch_size, end)) for i in range(start, end, batch_size)]

    return batch_idxs


def reorder_blocks(arr):
    """Helper function to reorder blocks walking out from the center"""
    n = len(arr)
    mid = (n - 1) // 2

    reordered = [arr[mid]]

    left = mid - 1
    right = mid + 1

    while left >= 0 or right < n:
        if right < n:
            reordered.append(arr[right])
            right += 1
        if left >= 0:
            reordered.append(arr[left])
            left -= 1

    return reordered


def calc_memmap_mean(file, frame_range, batch_size):
    """Helper function to calculate the mean over an arbitrary number of frames"""
    # Get the batch indexes
    start = frame_range[0]
    end = frame_range[1]
    step = batch_size

    batch_idxs = [(i, min(i + step, end)) for i in range(start, end, step)]

    mean_list = []
    for idx in batch_idxs:
        temp_frames = file[idx[0] : idx[1]]
        mean_list.append(np.nanmean(temp_frames, axis=0))

    mean_list = np.array(mean_list)

    mean = np.nanmean(mean_list, axis=0)

    return mean


def highpass_mean_image(I, aspect=1.0):
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
