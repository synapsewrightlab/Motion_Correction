import numpy as np
import torch
import time

import logging

from .patchwarp_rigid_pyramid import patchwarp_rigid_pyramid
from .patchwarp_rigid_cv import patchwarp_rigid_cv
from .patchwarp_save_tiff import patchwarp_save_tiff
from ..io.assign_reg_io import assign_reg_io

from ..suite2p_registration.make_summary import make_summary_video

logger = logging.getLogger(__name__)

from .. import parameters


def patchwarp_wrapper(
    f_raw_chan1,
    f_reg_chan1,
    f_raw_chan2,
    f_reg_chan2=None,
    refImg=None,
    align_by_chan2=False,
    save_path=None,
    settings=parameters.default_settings()["Patchwarp_settings"],
    save_tif=False,
    downsampled_frames=50,
    device=torch.device("cpu"),
):
    """
    Main registration function for patchwarp registration. For single and two channel images

    It will handle running the rigid and nonrigid motion correction for each imaging channel
    provided

    PARAMETERS
        f_raw_chan1: np.ndarray or BinaryFile
            Raw functional channel frames of shape (n_frmaes, Ly, Lx)

        f_reg_chan1: np.ndarray or BinaryFile
            Registered functional channel frames of shape (n_frmaes, Ly, Lx)

        f_raw_chan2: np.ndarray or BinaryFile
            Raw second channel

        f_reg_chan2: np.ndarray or BinaryFile
            Registered second channel

        refImg: np.ndarray or None
            Reference image of shape (Ly, Lx), dtype int16. If none, reference is computed
            from the data

        align_by_chan2: bool
            If True, use the second channel as the alignement source

        save_path: str or None
            Base directory for saving registered tiff files

        settings: dict
            Registration settings dictionary

        device: torch.device
            Torch device from computation

    OUTPUTS
        reg_outputs : dict


    """

    # Get the input outputs for registration
    # Get the input outputs for registration
    f_out = assign_reg_io(
        f_raw_chan1,
        f_reg_chan1,
        f_raw_chan2,
        f_reg_chan2,
        align_by_chan2,
        save_path,
        save_tif,
    )
    # Unpack i/o
    f_align_in, f_align_out, f_alt_in, f_alt_out, tif_align_out, tif_alt_out = f_out

    nchannels = 2 if f_alt_in is not None else 1
    logger.info(f"Registering {nchannels} channels")

    # Check to run rigid registration
    if settings["do_rigid"]:
        logger.info(f"Performing rigid registration")
        t_rigid = time.time()
        rigid_method = settings["rigid_method"]
        if rigid_method == "pyramid":
            logger.info("NOTE: Performing pyaramid based rigid registration")
            rigid_outputs = patchwarp_rigid_pyramid(
                f_align_in=f_align_in,
                f_align_out=f_align_out,
                f_alt_in=f_alt_in,
                f_alt_out=f_alt_out,
                pre_refImg=refImg,
                n_ch=nchannels,
                batch_size=settings["batch_size"],
                rigid_template_block_size=settings["rigid_template_block_size"],
                rigid_template_file_num=settings["rigid_template_frame_num"],
                pyramid_norm_method=settings["pyramid_norm_method"],
                pyramid_norm_radius=settings["pyramid_norm_radius"],
                pyramid_template_threshold=settings["pyramid_template_threshold"],
                pyramid_template_center_frac=settings["pyramid_template_center_frac"],
                pyramid_template_fftdenoise=settings["pyramid_template_fftdenoise"],
            )

        elif rigid_method == "cv":
            logger.info("NOTE: Performing OpenCV based rigid registration")
            rigid_outputs = patchwarp_rigid_cv(
                f_align_in=f_align_in,
                f_align_out=f_align_out,
                f_alt_in=f_alt_in,
                f_alt_out=f_alt_out,
                pre_refImg=refImg,
                n_ch=nchannels,
                batch_size=settings["batch_size"],
                rigid_template_block_size=settings["rigid_template_block_size"],
                rigid_template_file_num=settings["rigid_template_frame_num"],
                factor=settings["cv_factor"],
                marginh=settings["cv_marginh"],
                marginw=settings["cv_marginw"],
                sigma_smoothing=settings["cv_sigma_smoothing"],
                sigma_normalization=settings["cv_sigma_normalization"],
                normalization_offset=settings["cv_normalization_offset"],
                to_equalize_histogram=settings["to_equalize_histogram"],
                cv_corr_threshold=settings["cv_corr_threshold"],
            )
        logger.info(
            "-- Rigid registration complete, %0.2f sec." % (time.time() - t_rigid)
        )
    else:
        logger.info(f"Skipping rigid registration")
        rigid_outputs = {
            "refImg": None,
            "meanImg": None,
            "meanImg_chan2": None,
            "yoff": None,
            "xoff": None,
            "corrXY": None,
            "yrange": None,
            "xrange": None,
        }

    # Check to run non-rigid registration
    if settings["do_nonrigid"]:
        logger.info("STILL NEED TO WRITE NON-RIGID REGISTRATION")
        reg_outputs = rigid_outputs

    else:
        logger.info(f"Skipping nonrigid registration")
        reg_outputs = rigid_outputs

    # Output all the tiff files if specified
    if tif_align_out:
        logger.info(f"-- Saving aligned images as tif files")
        patchwarp_save_tiff(f_align_out, tif_align_out, settings["batch_size"])

    if tif_alt_out:
        patchwarp_save_tiff(f_alt_out, tif_alt_out, settings["batch_size"])

    # Make the summary output
    logger.info(
        f"---- Making the summary tif video downsampled by {downsampled_frames}"
    )
    summary_time = time.time()
    make_summary_video(f_align_out, downsampled_frames=downsampled_frames)
    if nchannels > 1:
        make_summary_video(f_alt_out, downsampled_frames=downsampled_frames)

    logger.info(
        "---- Done making summary video, %0.2f sec." % (time.time() - summary_time)
    )

    return reg_outputs
