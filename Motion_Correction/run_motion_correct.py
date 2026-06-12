import os
import time

import torch
import logging

from importlib.metadata import version
from pathlib import Path

from . import io

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

    save_path = database["save_path0"]

    logger.info(version("Motion_Correction"))
    logger.info(f"data_path: {database["data_path"]}")

    # Set up the databases for each plane
    database["input_format"] = database.get("input_format", "tif")
    databases = io.io_utils.get_file_list(database)
    

    




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