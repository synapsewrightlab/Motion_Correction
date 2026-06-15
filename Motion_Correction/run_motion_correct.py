import os
import time

import torch
import numpy as np
import logging
import contextlib

from importlib.metadata import version
from datetime import datetime
from pathlib import Path

from . import io, parameters, pipeline

logger = logging.getLogger(__name__)


def run_motion_correction(database={}, settings={}):
    """
    Run the motion correction pipeline across all planes and channels

    Converts input files to binary format, then runs the desired motion 
    correction alogrithm on each plane sequentially

    PARAMETERS
        database: dict
            Database dictionary specitying the "data_path", "nplanes", "nchannels",
            and other input/output configurations
        
        settings: dict
            Settings dictionary used by tyhe motion correction algorithms

    """
    start_time = time.time()

    database = {**parameters.default_database(), **database}
    settings = {**parameters.default_settings(), **settings}

    save_path = database["save_path0"]

    logger.info(version("Motion_Correction"))
    logger.info(f"data_path: {database["data_path"]}")

    # Set up the databases for each plane
    database["input_format"] = database.get("input_format", "tif")
    databases = io.io_utils.get_file_list(database)

    db_paths = [db["db_path"] for db in databases]
    settings_path = [db["settings_path"] for db in databases]
    save_folder = os.path.join(database["save_path0"], database["save_folder"])
    np.save(os.path.join(save_folder, "db.npy"), database)
    np.save(os.path.join(save_folder, "settings.npy"), settings)

    # Convert raw images to binary files
    databases = io.prep_image_files.prep_image_files(databases, settings)

    logger.info("Wrote {} frames per binary, {} folders + {} channels, {:0.2f}sec".format(
                databases[0]["nframes"], len(databases), databases[0]["nchannels"], 
                time.time() - start_time)
                )
    
    # Prepare the run each plane
    for ipl, (settings_path, db_path) in enumerate(zip(settings_path, db_paths)):
        ops = np.load(settings_path, allow_pickle=True).item()
        ops = {**ops, **settings}
        db = np.load(db_path, allow_pickle=True).item()
        logger.info(f"========================= PLANE {ipl} =========================")

        # Run the plane



def run_plane(database, settings):
    """
    Run the motion correction pipeline across all planes and channels

    Converts input files to binary format, then runs the desired motion 
    correction alogrithm on each plane sequentially

    PARAMETERS
        database: dict
            Database dictionary specitying the "data_path", "nplanes", "nchannels",
            and other input/output configurations
        
        settings: dict
            Settings dictionary used by tyhe motion correction algorithms

    OUTPUTS
        outputs: dict
            Outputs from the motion correction. See X for details
    
    """
    settings = {**parameters.default_settings(), **settings}
    settings["date_proc"] = datetime.now().astimezone()

    # Set up torch device
    device = _assign_torch_device(settings["torch_device"])

    # Check for sufficient frame numbers
    if database["nframes"] < 50:
        raise ValueError("Number of frames should at least be 50")
    elif database["nframes"] < 200:
        logger.warning("WARNING: Number of frames < 200, unpredictable behavior")
    
    # Get the binary file paths
    raw_file_chan1 = database["raw_file_chan1"]
    reg_file_chan1 = database["reg_file_chan1"]
    two_ch = database["nchannels"] > 1
    raw_file_chan2 = database.get("raw_file_chan2", None)
    reg_file_chan2 = database.get("reg_file_chan2", None)

    # Get the shape of the binary files
    n_frames, Ly, Lx = database["nframes"], database["Ly"], database["Lx"]

    null = contextlib.nullcontext()
    # Load in the binary files
    with io.binary.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file_chan1, n_frames=n_frames, write=False) as f_raw_chan1, \
        io.binary.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file_chan1, n_frames=n_frames, write=True) as f_reg_chan1, \
        io.binary.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file_chan2, n_frames=n_frames, write=False) if two_ch else null as f_raw_chan2, \
        io.binary.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file_chan2, n_frames=n_frames, write=True) if two_ch else null as f_reg_chan2:

        # Run the motion correction pipeline
        pass




def logger_setup(save_path=None):
    """
    Configure logging for the motion correction package

    Set up console and file logging handlers for the motion correction logger

    PARAMETERS
        save_path: str, optional (default=None)
            Directory to write the log file. If none, only console logging is done.

    """
    if save_path is not None and not Path(save_path).exists():
        Path(save_path).mkdir(parents=True, exist_ok=True)

    mc_logger = logging.getLogger("motion_correction")
    mc_logger.setLevel(logging.DEBUG)

    if not mc_logger.handlers:
        # Add console handler at info level with shorter messages
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console.setFormatter(file_formatter)
        mc_logger.addHandler(console)
    
    if save_path is not None:
        log_file = Path(save_path) / "run.log"
        try:
            log_file.unlink()
        except:
            pass
        print(f"creating log file {log_file}")
        file = logging.FileHandler(log_file, mode='w')
        file.setLevel(logging.DEBUG)
        log_file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file.setFormatter(log_file_formatter)
        mc_logger.addHandler(file)


def _assign_torch_device(str_device):
    """Validate and return a torch device"""
    if str_device == "cpu":
        logger.info("** Using CPU **")
        return torch.device(str_device)
    else:
        try:
            device = torch.device(str_device)
            _ = torch.zeros([1,2,3]).to(device)
            logger.info(f"** torch.device('{str_device}') installed and working **")
            return torch.device(str_device)
        except:
            logger.info(f"torch.device('{str_device}') not working, using cpu")
            return torch.device("cpu")