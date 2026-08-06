import numpy as np


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
