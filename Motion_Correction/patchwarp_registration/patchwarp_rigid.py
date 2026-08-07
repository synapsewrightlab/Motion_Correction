import os
import time

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging

from . import patchwarp_utils as utils
from .make_reference_image import make_reference_image
from .pyramid_registration import pyramid_parallel

logger = logging.getLogger(__name__)


def patchwarp_rigid(
    f_align_in,
    f_align_out,
    f_alt_in,
    f_alt_out,
    pre_refImg,
    n_ch,
    batch_size,
    rigid_norm_method,
    rigid_norm_radius,
    rigid_template_block_size,
    rigid_template_threshold,
    rigid_template_tiffstack_num,
    rigid_template_center_frac,
    rigid_template_fftdenoise,
):
    """
    Function to perform rigid motion correction. Based on the method developed by Akinori Mitani


    PARAMETERS
        f_align_in : np.ndarray or BinaryFile
            Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.

        f_align_out: np.ndarray or BinaryFile
            Registered functional channel frames of shape (n_frmaes, Ly, Lx)

        f_alt_in : np.ndarray or BinaryFile
            Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.
            For the other channel not used to compute shifts

        f_alt_out : np.ndarray or BinaryFile
            Registered functional channel frames of shape (n_frmaes, Ly, Lx)
            For the other channel not used to compute shifts

        pre_refImg : np.ndarray
            Optional: Pre-computed reference image (Ly, Lx)

        n_ch : int
            Number of channels being registered

        batch_size : int
            Number of frames per batch - choose fewer if using GPU and running out of memory.

        rigid_norm_method : str
            Normalization method. Can be 'rank' or 'local'

        rigid_norm_radius : int
            Radius of normalization filter for the local method

        rigid_template_block_size : int
            Number of image blocks per imaging session to split for rigid correction

        rigid_template_threshold : float
            Quantile threshold for selecting frames used for template image

        rigid_template_tiffstack_num : int
            Number of image stacks (files) to estimate template images

        rigid_template_center_frac : float
            Central fraction of template used for registration

        rigid_template_fftdenoise : bool
            Remove strong period noise (i.e., ripple PMT noise) from the template image

    OUTPUT



    """
    nframes, Ly, Lx = f_align_in.shape

    # Check there are enough frames to estimate the reference image
    if nframes < rigid_template_tiffstack_num:
        logger.info(
            f"WARNING: Rigid_template_tiffstack_num is larger than number of image frames"
        )
        rigid_template_tiffstack_num = nframes
        logger.info(
            f"-------- Rigid_template_tiffstack_num has been changed to {rigid_template_tiffstack_num}"
        )

    # Set up the blocks
    ## Force odd number of blocks
    if rigid_template_block_size % 2 == 0:
        rigid_template_block_size += 1

    # Get block Ids
    block_idxs = utils.get_block_ids(nframes, rigid_template_block_size)
    ordered_blocks = utils.reorder_blocks(block_idxs)

    # Make the initial template from central block
    if pre_refImg is None:
        initial_frames = ordered_blocks[0]
        # Get the central frames
        center_frame = (initial_frames[0] + initial_frames[1]) // 2
        logger.info(
            f"Making initial reference image from frames centered on {center_frame}"
        )
        t_ref = time.time()
        initial_start = center_frame - int((rigid_template_tiffstack_num - 1) / 2)
        initial_end = center_frame + int(rigid_template_tiffstack_num / 2)

        # make the initial reference image
        initial_frames = f_align_in[initial_start:initial_end]
        initial_reference_image = make_reference_image(
            initial_frames,
            threshold=rigid_template_threshold,
            fftdenoise=rigid_template_fftdenoise,
            rigid_norm_method=rigid_norm_method,
            rigid_norm_radius=rigid_norm_radius,
            rigid_template_center_frac=rigid_template_center_frac,
        )
        logger.info("-- Reference image generated, %0.2f sec." % (time.time() - t_ref))
    else:
        logger.info("Using pre-computed reference image")
        initial_reference_image = pre_refImg

    # Get all the offsets
    xoff = np.zeors(nframes)
    yoff = np.zeros(nframes)
    corrXY = np.zeros(nframes)

    # Store copies of reference images and mean aligned images
    refImg = [None for _ in range(len(ordered_blocks))]
    meanImg = [None for _ in range(len(ordered_blocks))]
    meanImg_chan2 = [None for _ in range(len(ordered_blocks))]

    # Iterate through the blocks
    for iblock, block in enumerate(ordered_blocks):
        block_time = time.time()
        # Use the initial reference for the first block
        if iblock == 0:
            reference_image = initial_reference_image
        # Use previous block's frames for reference image if second block
        elif iblock == 1:
            ## Get the previous frames from the last block to make the reference
            ref_end = ordered_blocks[iblock - 1][1]
            ref_start = ref_end - rigid_template_tiffstack_num - 1
            reference_image = np.nanmean(f_align_out[ref_start, ref_end], axis=0)
        ## Go two blocks back to get the adjacent blocks frames for reference
        else:
            ## Check if block is going backwards from the center
            if iblock % 2 == 0:
                ref_start = ordered_blocks[iblock - 2][0]
                ref_end = ref_start + rigid_template_tiffstack_num - 1
                reference_image = np.nanmean(f_align_out[ref_start, ref_end], axis=0)
            else:
                ref_end = ordered_blocks[iblock - 2][1]
                ref_start = ref_end - rigid_template_tiffstack_num - 1
                reference_image = np.nanmean(f_align_out[ref_start, ref_end], axis=0)

        orig_block_idx = block_idxs.index(block)
        refImg[orig_block_idx] = reference_image

        # Set up the batch idxs
        start = block[0]
        end = block[1]
        step = batch_size
        batch_idxs = [(i, min(i + step, end)) for i in range(start, end, step)]

        # Prepare to parallize across batches
        ## Grab relevant memmap info
        input_path = f_align_in.filename
        input_shape = f_align_in.shape
        output_path = f_align_out.filename
        output_shape = f_align_out.shape

        if n_ch > 1:
            input_path_alt = f_alt_in.filename
            input_shape_alt = f_alt_in.shape
            output_path_alt = f_alt_out.filename
            output_shape_alt = f_alt_out.shape
        else:
            input_path_alt = None
            input_shape_alt = None
            output_path_alt = None
            output_shape_alt = None

        # Change if you want to max out cpus
        n_workers = os.cpu_count() - 2
        pyramid_args = [
            rigid_norm_method,
            rigid_norm_radius,
            rigid_template_center_frac,
        ]

        jobs = [
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
            )
            for start, end in batch_idxs
        ]

        logger.info(
            f"--- Performing pyramid registration on block {iblock} with {n_workers} workers"
        )

        temp_mean = []
        temp_mean_alt = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(pyramid_parallel, job): job for job in jobs}
            for future in as_completed(futures):
                y, x, c, mean, mean_alt = future.result()
                logger.info(f"------ Finished with frames {start}-{end}")
                xoff[start:end] = x
                yoff[start:end] = y
                corrXY[start:end] = c
                temp_mean.append(mean)
                temp_mean_alt.append(mean_alt)

        block_mean = np.nanmean(temp_mean, axis=0)
        if temp_mean_alt[0]:
            block_mean_alt = np.nanmean(temp_mean_alt, axis=0)
        else:
            block_mean_alt = None

        meanImg[orig_block_idx] = block_mean
        meanImg_chan2[orig_block_idx] = block_mean_alt

        logger.info(
            f"--- Block {iblock} registration complete, {time.time() - block_time:.2f} sec."
        )

    # Calculate yrange and xrange
    ymin = np.ceil(np.abs(yoff).max())
    xmin = np.ceil(np.abs(xoff).max())
    ymax = Ly - ymin
    xmax = Lx - ymax

    yrange = [int(ymin), int(ymax)]
    xrange = [int(xmin), int(xmax)]

    # Normalize and reshape the reference and mean images
    refImg = np.array([utils.highpass_mean_image(x, 1.0) for x in refImg])
    meanImg = np.array([utils.highpass_mean_image(x, 1.0) for x in meanImg])
    if n_ch > 1:
        meanImg_chan2 = np.array(
            [utils.highpass_mean_image(x, 1.0) for x in meanImg_chan2]
        )

    # Prepare the registration outputs
    rigid_outputs = {
        "refImg": refImg,
        "meanImg": meanImg,
        "meanImg_chan2": meanImg_chan2,
        "yoff": yoff,
        "xoff": xoff,
        "corrXY": corrXY,
        "yrange": yrange,
        "xrange": xrange,
    }

    return rigid_outputs
