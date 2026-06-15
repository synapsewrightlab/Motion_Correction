import glob
import os
import copy
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

import numpy as np
from natsort import natsorted

EXTS = {"tif": ["*.tif", "*.tiff", "*.TIF", "*.TIFF"],
        "h5": ["*.h5", "*.hdf5", "*.mesc"],
        "sbx": ["*.sbx"],
        "nd2": ["*.nd2"],
        "dcimg": ["*.dcimg"],
        "bruker": ["*.ome.tif", "*.ome.TIF"],
        "movie": ["*.mp4", "*.avi"]}

def list_files(file_root, exts, one_down=False):
    """Collect files matching given extension
    
    INPUT PARAMETERS
        file_root: str
            Root directory to search for files
        
        exts: list of str
            Glob patterns to match for file extensions
        
        one_down: bool
            If True, will also search subdirectories of root directory
    
    OUTPUT
        file_list: list of str
            Naturally sorted list of matching file paths
        
        first_files: np.ndarray
            Boolean array of length len(file_list), where True marks the first
            file from each folder to track folder boundaries
    
    """
    # Initialize outputs
    file_list = []
    first_files = np.zeros(0, "bool")
    # Find all files with the extension in the root directory
    for e in exts:
        file_path = os.path.join(file_root, e)
        file_list.extend(glob.glob(file_path))
    # Naturally sort the files
    file_list = natsorted(set(file_list))
    # Demarcate the first file in the folder
    if len(file_list) > 0:
        first_files = np.zeros(len(file_list), "bool")
        first_files[0] = True
    # Get the length of the file list
    file_list_length = len(file_list)
    # Repeat in subdirectories if specified
    if one_down:
        f_dir = natsorted(glob.glob(os.path.join(file_root, "*/")))
        for folder_down in f_dir:
            new_file_list = []
            for e in exts:
                file_path = os.path.join(folder_down, e)
                new_file_list.extend(glob.glog(file_path))
            new_file_list = natsorted(set(new_file_list))
            # Demarcate first file in folder and update file list
            if len(new_file_list) > 0:
                file_list.extend(new_file_list)
                first_files = np.append(first_files, np.zeros((len(new_file_list),), "bool"))
                first_files[file_list_length] = True
                file_list_length = len(file_list)

    return file_list, first_files


def get_file_list(database):
    """
    Build tthe list of input files from the database configuration

    INPUT PARAMETERS
        database: dict
            Database dictionary. Must contain the "data_path" (list of str). Optionally,
            contains "file_list" (list of str), "subfolders" (list of str), "one_down" (bool),
            and "input_format" (str, default "tif")

    OUTPUT
        all_files: list of str
            List of all file paths
        
        first_files: np.ndarray
            Boolean array of length len(all_files), where True marks the first file
            from each folder
    
    """
    # Get data path and file type
    data_path = database["data_path"]
    input_format = database.get("input_format", "tif")

    # use a user specified list of files
    if database.get("file_list", None) is not None:
        all_files = []
        for file in database["file_list"]:
            all_files.append(os.path.join(data_path[0], file))
        first_files = np.zeros(len(all_files), dtype="bool")
        first_files[0]= True
        logger.info(f"--- Found {len(all_files)} files - converting to binary")
    
    # Find files if not explicity given by user
    else:
        # Add subfolders to search if specified
        if len(data_path) == 1 and database.get("subfolders", None) is not None:
            folder_list = []
            for folder_down in database["subfolders"]:
                folder = os.path.join(data_path[0], folder_down)
                folder_list.append(folder)
        else:
            folder_list = data_path

        all_files = []
        first_files = []

        for k, folder in enumerate(folder_list):
            files, firsts = list_files(
                folder,
                database["one_down"],
                EXTS[input_format]
            )
            all_files.extend(files)
            first_files.extend(list(firsts))
        
        if len(all_files) == 0:
            logger.info(f"Could not find any {EXTS[input_format]} files in {data_path}")
            raise Exception("No Files Found")

        else:
            first_files = np.array(first_files).astype("bool")
            logger.info(f"--- Found {len(all_files)} files - converting to binary")
    
    return all_files, first_files


def init_database(db0):
    """
    Initialize per-plane database dictionaries and create output dictionaries

    Creates a deep copy of db0 for each plane and each ROI, setting up save paths, 
    and fast-disk directories

    INPUT PARAMETERS
        db0: dict
            Base database dictionary. Must contain "nplanes", "nchannels", 
            "save_path0", and "save_folder". Optionally contains "fast_disk",
            "iplane", "lines", "dy", and "dx" (for multiple ROIs)

        OUTPUT
            databases: list of dict
                List of per-plane database dictionaries, each with added keys
                "save_path", "fast_dict", "settings_path", "db_path", "reg_file",
                and optionally "reg_file_chan2", "lines", "dy", "dx", "iroi",
                "iplane"
    
    """
    nplanes = db0["nplanes"]
    nchannels = db0["nchannels"]
    nfolders = nplanes
    iplane = db0.get("iplane", np.arange(0, nplanes))
    has_lines = False

    # Deal with multiple rois if specified
    if "lines" in db0 and db0["lines"] is not None and len(db0["lines"]) > 0:
        nrois = len(db0["lines"])
        db0["nrois"] = nrois
        nfolders *= nrois
        logger.info(f"NOTE: nplanes = {nplanes}, nrois = {nrois} -> nfolders = {nfolders}")
        # Replicate lines across planes if nplanes > 1
        if nplanes > 1:
            lines0, dy0, dx0 = db0["lines"].copy(), db0["dy"].copy(), db0["dx"].copy()
            dy0, dx0 = np.array(dy0), np.array(dx0)
            dy = np.tile(dy0[np.newaxis, :], (nplanes, 1)).flatten()
            dx = np.tile(dx0[np.newaxis, :], (nplanes, 1)).flatten()
            lines = []
            [lines.extend(lines0) for _ in range(nplanes)]
            iroi = np.tile(np.arange(nrois)[np.newaxis,:], (nplanes, 1)).flatten()
            iplane = np.tile(np.arange(nplanes)[:, np.newaxis], (1, nrois)).flatten()
        else:
            lines, dy, dx = db0["lines"].copy(), db0["dy"].copy(), db0["dx"].copy()
            iroi = np.arange(nrois)
            iplane = np.zeros(nrois, "int")
        has_lines = True 

    # Initialize database list
    database_list = []

    # Set up the fast disk 
    if db0.get("fast_disk", None) is None or len(db0["fast_disk"]) == 0:
        db0["fast_disk"] = db0["save_path0"]
    
    fast_disk = db0["fast_disk"]

    # Compile databases into list across the different planes
    for j in range(0, nfolders):
        db = copy.deepcopy(db0)
        db["save_path"] = os.path.join(db["save_path0"], db["save_folder"], f"plane{j}")
        fast_disk = os.path.join(db["fast_disk"], "temp", f"plane{j}")
        db["fast_disk"] = fast_disk
        db["settings_path"] = os.path.join(db["save_path"], "settings.npy")
        db["db_path"] = os.path.join(db["save_path"], "database.npy")

        db["save_path_chan1"] = os.path.join(db["save_path"], "Ch1")
        db["raw_file_chan1"] = os.path.join(fast_disk, "raw_chan1.bin")
        db["reg_file_chan1"] = os.path.join(db["save_path_chan1"], "aligned_chan1.bin")

        if has_lines:
            db["lines"], db["dy"], db["dx"] = lines[j], dy[j], dx[j]
            db["iroi"] = iroi[j]
        db["iplane"] = iplane[j]
        if nchannels > 1:
            db["save_path_chan2"] = os.path.join(db["save_path"], "Ch2")
            db["raw_file_chan2"] = os.path.join(fast_disk, "raw_chan2.bin")
            db["reg_file_chan2"] = os.path.join(db["save_path_chan2"], "aligned_chan2.bin")
        
        # Make sure paths exist
        os.makedirs(db["fast_disk"], exist_ok=True)
        os.makedirs(db["save_path_chan1"], exist_ok=True)
        if nchannels > 1:
            os.makedirs(db["save_path_chan2"], exist_ok=True)
        database_list.append(db)

    return database_list


