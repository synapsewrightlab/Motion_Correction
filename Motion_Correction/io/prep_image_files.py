import os
import logging
import contextlib

from . import tiff_files

logger = logging.getLogger(__name__)

files_to_binary = {
    "tif": tiff_files.tiff_to_binary,
    "bruker": tiff_files.ome_to_binary,
}

def prep_image_files(databases, settings):
    """Function to handle the initial conversion of image files 
    to binary
    
    INPUT PARAMETERS
        databases: list of dict
            Database dictionaries for each plane/ROI. Must contain  keys "file_list",
            "first_files", "batch_size", "force_sktiff", "nplanes", "nchannels", and
            optionally "nrois", "swap_order", "lines". Updated in-place with "Ly", "Lx",
            "nframes", "frames_per_file", "frames_per_folder", "meanImg", and
            "meanImg_chan2".

        settings : dict
            Suite2p settings dictionary, saved alongside each plane's database.

    OUTPUT
        databases: list of dict
            Updated database dictionaries

    """
    # Get input format
    input_format = databases[0]["input_format"]

    # Set up the binary files to write to
    with contextlib.ExitStack() as stack:
        # Set up the file names
        fnames = [db["raw_file_chan1"] for db in databases]
        files = [stack.enter_context(open(f, "wb")) for f in fnames]
        if databases[0]["nchannels"] > 1:
            fnames_ch2 = [db["raw_file_chan2"] for db in databases]
            files_ch2 = [stack.enter_context(open(f2, "wb")) for f2 in fnames_ch2]
        else:
            files_ch2 = None
        
        # Convert the binary
        databases = files_to_binary[input_format](databases, settings, files, files_ch2)

    return databases



    