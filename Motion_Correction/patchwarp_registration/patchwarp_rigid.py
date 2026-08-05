import numpy as np
import torch

import logging

from .make_template import make_template

logger = logging.getLogger(__name__)


def patchwarp_rigid(
    f_align_in,
    f_align_out,
    f_alt_in,
    f_alt_out,
    n_ch,
    rigid_norm_method,
    rigid_norm_radius,
    rigid_template_block_size,
    rigid_template_threshold,
    rigid_template_tiffstack_num,
    rigid_template_center_frac,
    rigid_template_fftdenoise,
    device=torch.device("cpu"),
    tif_root_align=None,
    tif_root_atl=None,
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

        f_alt_out: np.ndarray or BinaryFile
            Registered functional channel frames of shape (n_frmaes, Ly, Lx)
            For the other channel not used to compute shifts

        n_ch : int
            Number of channels being registered

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

        device: torch.device
            Torch device from computation

        tif_root_align : str or None
            If provided, save registered frames as tiffs in this directory.
            For the align channel

        tif_root_alt : str or None
            If provided, save registered frames as tiffs in this directory.
            For the other channel channel

    OUTPUT



    """
    nframes, Ly, Lx = f_align_in.shape

    # Check there are enough frames to estimate the reference image
    if nframes < rigid_template_tiffstack_num:
        logger.info(
            f"rigid_template_tiffstack_num is larger than number of image frames"
        )
        rigid_template_tiffstack_num = nframes
        logger.info(
            f"rigid_template_tiffstack_num has been changed to {rigid_template_tiffstack_num}"
        )

    # Find middle file that we will move out from to define reference
    middle_file_id = int(nframes / 2)

    # Make initial reference image
    logger.info(f"Making Reference Image from frames centered on {middle_file_id}")
    start = middle_file_id - int((rigid_template_tiffstack_num - 1) / 2)
    end = middle_file_id + int(rigid_template_tiffstack_num / 2)
    frames = f_align_in[start:end]
    reference_image = make_template(
        frames, rigid_template_threshold, rigid_template_fftdenoise
    )

    # Register frames to template before recalculating template
