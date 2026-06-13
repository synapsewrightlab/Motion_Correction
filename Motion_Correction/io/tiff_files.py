import gc
import glob
import json
import math
import os
import time
import logging

logger = logging.getLogger(__name__)

import numpy as np
from tifffile import imread, TiffFile, TiffWriter

from . import io_utils

# Check for scanimaage tiff reader
try:
    from ScanImageTiffReader import ScanImageTiffReader
    HAS_SCANIMAGE = True

except ImportError:
    HAS_SCANIMAGE = False


def open_tiff(file, sktiff):
    """
    Open tiff file and return the reader object and number of pages

    Uses either tifffile or ScanImageTiffReader dependingon sktiff

    INPUT PARAMETERS
        file: str
            Path to the tiff file
        
        sktiff: bool
            If True, uses tifffile. If False, uses ScanImageTiffReader

    OUTPUTS
        tif: TiffFile or ScanImageTiffReader
            Opened TIFF reader object

        Ltif: int
            Number of pages (frames) in the TIFF file
    
    """

    if sktiff:
        tif = TiffFile(file)
        Ltif = len(tif.pages)

    else:
        tif = ScanImageTiffReader(file)
        Ltif = 1 if len(tif.shape()) < 3 else tif.shape()[0]  # single page tiffs

    return tif, Ltif


def use_sktiff_reader(tiff_filename, batch_size):
    """
    Test whether ScanImageTiffReader can read a TIFF file.

    Attempts to read a small batch from the file using ScanImageTiffReader. If it
    succeeds, returns False (use ScanImageTiffReader). If it fails or the package is
    not installed, returns True (fall back to tifffile).

    INPUT PARAMETERS
        tiff_filename : str
            Path to the TIFF file to test.

        batch_size : int
            Number of frames to attempt reading as a test.

    OUTPUT
        use_sktiff : bool
            True if tifffile should be used, False if ScanImageTiffReader works.
    """
    if HAS_SCANIMAGE:
        try:
            with ScanImageTiffReader(tiff_filename) as tif:
                tif.data() if len(tif.shape()) < 3 else tif.data(
                    beg=0, end=np.minimum(batch_size,
                                          tif.shape()[0] - 1))
            return False
        
        except:
            logger.info(
                "NOTE: ScanImageTiffReader not working for this tiff type, using tifffile"
            )
            return True
        
    else:
        logger.info("NOTE: ScanImageTiffReader not installed, using tifffile")
        return True
    

def read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff):
    """
    Read a batch of frames from an open TIFF file starting at index 'ix'

    Reads up to 'batch_size' frames and converts data into int16. Returns None
    if 'ix' is past the end of the file

    PARAMETERS
        file : str
            Path to the TIFF file (used by tifffile's `imread` when `use_sktiff` is True).

        tif : TiffFile or ScanImageTiffReader
            Already-opened TIFF reader object.

        Ltif : int
            Total number of pages (frames) in the TIFF file.

        ix : int
            Starting frame index to read from.

        batch_size : int
            Maximum number of frames to read in this batch.

        use_sktiff : bool
            If True, read with tifffile (`imread`). If False, read with ScanImageTiffReader.

    OUTPUT
        im: np.ndarray or None
            Frames as an int16 array of shape (num_frames, Ly, Lx)

    """

    if ix >= Ltif:
        return None
    nfr = min(Ltif - ix, batch_size)
    if use_sktiff:
        im = imread(file, key=range(ix, ix + nfr))
    elif Ltif == 1:
        im = tif.data()
    else:
        im = tif.data(beg=ix, end=ix + nfr)
    # for single-page tiffs, add 1st dim
    if len(im.shape) < 3:
        im = np.expand_dims(im, axis=0)

    # check if uint16
    if im.dtype.type == np.uint16:
        im = (im // 2).astype(np.int16)
    elif im.dtype.type == np.int32:
        im = (im // 2).astype(np.int16)
    elif im.dtype.type != np.int16:
        im = im.astype(np.int16)

    if im.shape[0] > nfr:
        im = im[:nfr, :, :]

    return im


def tiff_to_binary(databases, settings, reg_file, reg_file_chan2):
    """
    Read TIFF files and write interleaved plane and channel data to binary files

    Iterates over all TIFF files listed in databases[0]["file_list], de-interleaves
    planes and channels, and writes each plane's frames to the corresponding binary
    file. Also computes per-plane mean images and frame counts

    INPUT PARAMETERS
        databases: list of dict
            Database dictionaries for each plane/ROI. Must contain  keys "file_list",
            "first_files", "batch_size", "force_sktiff", "nplanes", "nchannels", and
            optionally "nrois", "swap_order", "lines". Updated in-place with "Ly", "Lx",
            "nframes", "frames_per_file", "frames_per_folder", "meanImg", and
            "meanImg_chan2".

        settings : dict
            Suite2p settings dictionary, saved alongside each plane's database.

        reg_file : list of file objects
            Opened binary files for writing each plane's functional channel data.

        reg_file_chan2 : list of file objects
            Opened binary files for writing each plane's second channel data
            (used only when nchannels > 1).

    OUTPUT
        databases: list of dict
            Updated database dictionaries with image dimensions, frame coutns, and mean images
            populated
    
    """
    start_time = time.time()

    files = databases[0]["file_list"]
    first_files = databases[0]["first_files"]

    # try tiff readers
    batch_size = databases[0]["batch_size"]
    use_sktiff = True if databases[0]["force_sktiff"] else use_sktiff_reader(files[0], batch_size=batch_size)

    nplanes, nchannels = databases[0]["nplanes"], databases[0]["nchannels"]

    # Processing for multiple rois
    nrois = databases[0].get("nrois", 1)
    batch_size = nplanes * nchannels * math.ceil(batch_size / (nplanes * nchannels))

    # loop over all tiffs
    which_folder = -1
    ntotal = 0
    swap = databases[0].get("swap_order", False)

    for i, file in enumerate(files):
        # open the tif
        tif, Ltif = open_tiff(file, use_sktiff)
        # Keep track of the plane identify of the first frame (channel identify is assumed always 0)
        if first_files[i]:
            which_folder += 1
            iplane = 0
        ix = 0
        while 1:
            # Read in the current frame image
            im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
            # Break loop if the end of the tiff file
            if im is None:
                break          
            nframes = im.shape[0]
            # Iterate through each plane
            for j in range(0, nplanes):
                # Set up initial values if first image
                if i == 0 and ix == 0:
                    Ly, Lx = im.shape[1], im.shape[2]
                    for k in range(nrois):
                        jk = j*nrois + k
                        Ly = (databases[jk]["lines"][-1] + 1 - databases[jk]["lines"][0] 
                              if nrois > 1 else Ly)
                        databases[jk]["Ly"], databases[jk]["Lx"] = Ly, Lx
                        databases[jk]["nframes"] = 0
                        databases[jk]["frames_per_file"] = np.zeros(len(files), "int")
                        databases[jk]["frames_per_folder"] = np.zeros(first_files.sum(), "int")
                        databases[jk]["meanImg_chan1"] = np.zeros((Ly, Lx), "float64")
                        if nchannels > 1:
                            databases[jk]["meanImg_chan2"] = np.zeros((Ly, Lx), "float64")

                if not swap:
                    i0 = nchannels * ((iplane + j) % nplanes)
                else:
                    i0 = (iplane + j) % (nplanes*nchannels)

                # if nchannels > 1:
                #     nfunc = databases[jk]["functional_chan"] - 1
                # else:
                #     nfunc = 0

                # Hold over from Suit2p about saving information for the functional channel in the database
                nfunc = 0

                #print(i0, int(i0) + (swap+1)*nfunc)

                # Handle channel 1 first
                im2write = im[int(i0) + (swap+1)*nfunc:nframes:nplanes * nchannels]
                
                for k in range(nrois):
                    jk = j*nrois + k
                    if nrois > 1:
                        imk = im2write[:, databases[jk]["lines"][0] : databases[jk]["lines"][-1] + 1]
                    else:
                        imk = im2write
                    reg_file[jk].write(bytearray(imk))
                    databases[jk]["meanImg_chan1"] += imk.sum(axis=0).astype("float64")
                    databases[jk]["nframes"] += imk.shape[0]
                    databases[jk]["frames_per_file"][i] += imk.shape[0]
                    databases[jk]["frames_per_folder"][which_folder] += imk.shape[0]
                    
                if nchannels > 1:
                    #print(int(i0) + (swap+1)*(1 - nfunc))
                    im2write = im[int(i0) + (swap+1)*(1 - nfunc):nframes:nplanes * nchannels]
                    for k in range(nrois):
                        jk = j*nrois + k
                        if nrois > 1:
                            imk = im2write[:, databases[jk]["lines"][0] : databases[jk]["lines"][-1] + 1]
                        else:
                            imk = im2write
                        reg_file_chan2[jk].write(bytearray(imk))
                        databases[jk]["meanImg_chan2"] += imk.sum(axis=0).astype("float64")
            if not swap:
                iplane = (iplane - nframes / nchannels) % nplanes
            else:
                iplane = (iplane - nframes) % (nchannels * nplanes)
            ix += nframes
            ntotal += nframes
            if ntotal % (batch_size * 4) == 0:
                logger.info("%d frames of binary, time %0.2f sec." %
                      (ntotal, time.time() - start_time))
        gc.collect()

    # write dbs and settings files
    for db in databases:
        db["meanImg_chan1"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        np.save(db["save_path0"], db)
        np.save(db["save_path0"], settings)
    
    # close all binary files
    for jk in range(0, nplanes * nrois):
        reg_file[jk].close()
        if nchannels > 1:
            reg_file_chan2[jk].close()

    return databases


def ome_to_binary(databases, settings, reg_file, reg_file_chan2):
    """
    Convert non-interleaved OME-TIFF files to binary, splitting channels by filename

    Designed for recordings where channels are stored in separate TIFF files
    (distinguished by "Ch1"/"Ch2" in the filename) rather than interleaved pages.
    Supports both single-page and multi-page TIFFs, and handles bidirectional
    Bruker plane ordering.

    PARAMETERS
        databases: list of dict
            Database dictionaries for each plane. Must contain keys "file_list",
            "first_files", "batch_size", "nplanes", "nchannels", "functional_chan",
            and optionally "bruker_bidirectional". Updated in-place with "Ly", "Lx",
            "nframes", "frames_per_file", "frames_per_folder", "meanImg", and
            "meanImg_chan2".

        settings : dict
            Suite2p settings dictionary, saved alongside each plane's database.

        reg_file : list of file objects
            Opened binary files for writing each plane's functional channel data.

        reg_file_chan2 : list of file objects
            Opened binary files for writing each plane's second channel data
            (used only when nchannels > 1).

    OUTPUT
        databases : list of dict
            Updated database dictionaries with image dimensions, frame counts, and
            mean images populated.
    
    """
    start_time = time.time()

    nplanes = databases[0]["nplanes"]
    nchannels = databases[0]["nchannels"]

    # get file list from dbs (already populated by run_s2p.py)
    files = databases[0]["file_list"]
    first_files = databases[0]["first_files"]
    batch_size = databases[0]["batch_size"]
    use_sktiff = not HAS_SCANIMAGE


    files_Ch1, files_Ch2 = [], []
    for f in files:
        if f.find("Ch1") > -1:
            files_Ch1.append(f)
        else:
            files_Ch2.append(f)

    if len(files_Ch2) == 0:
        databases[0]["nchannels"] = 1
        nchannels = 1
    logger.info(f"nchannels = {nchannels}")

    # Set up the tif reader
    TiffReader = ScanImageTiffReader if HAS_SCANIMAGE else TiffFile
    with TiffReader(files_Ch1[0]) as tif:
        if HAS_SCANIMAGE:
            n_pages = tif.shape()[0] if len(tif.shape()) > 2 else 1
            shape = tif.shape()[-2:]
        else:
            n_pages = len(tif.pages)
            im0 = tif.pages[0].asarray()
            shape = im0.shape
        
    # Initialize new database variables
    for db in databases:
        db["nframes"] = 0
        db["frames_per_folder"] = np.zeros(first_files.sum(), "int")
        db["frames_per_file"] = np.ones(len(files_Ch1), "int") if n_pages==1 else np.zeros(len(files_Ch1), "int")
        db["meanImg_chan1"] = np.zeros(shape, np.float32)
        if nchannels > 1:
            db["meanImg_chan2"] = np.zeros(shape, np.float32)

    # Check for bidirection plan imaging
    bruker_bidirectional = databases[0].get("bruker_bidirectional", False)
    iplanes = np.arange(0, nplanes)

    # Index each file to a plane
    if not bruker_bidirectional:
        iplanes = np.tile(iplanes[np.newaxis, :],
                          int(np.ceil(len(files_Ch1) / nplanes))).flatten()
        iplanes = iplanes[:len(files_Ch1)]
    else:
        iplanes = np.hstack((iplanes, iplanes[::-1]))
        iplanes = np.tile(iplanes[np.newaxis, :],
                          int(np.ceil(len(files_Ch1) / (2 * nplanes)))).flatten()
        iplanes = iplanes[:len(files_Ch1)]

    # Iterate through files for channel 1
    itot = 0
    for ik, file in enumerate(files_Ch1):
        ip = iplanes[ik]
        # read tiff
        if n_pages==1:    
            with TiffReader(file) as tif:
                im = tif.data()  if HAS_SCANIMAGE else tif.pages[0].asarray()
            if im.dtype.type == np.uint16:
                im = (im // 2)
            im = im.astype(np.int16)

            # write to binary
            databases[ip]["nframes"] += 1
            databases[ip]["frames_per_folder"][0] += 1
            databases[ip]["meanImg_chan1"] += im.astype(np.float32)
            reg_file[ip].write(bytearray(im))

        else:
            tif, Ltif = open_tiff(file, not HAS_SCANIMAGE)
            # keep track of the plane identity of the first frame (channel identity is assumed always 0)
            ix = 0
            while 1:
                im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
                if im is None:
                    break
                nframes = im.shape[0]
                ix += nframes
                itot += nframes
                reg_file[ip].write(bytearray(im))
                databases[ip]["meanImg_chan1"] += im.astype(np.float32).sum(axis=0)
                databases[ip]["nframes"] += im.shape[0]
                databases[ip]["frames_per_file"][ik] += nframes
                databases[ip]["frames_per_folder"][0] += nframes
                if itot % 1000 == 0:
                    logger.info("%d frames of binary, time %0.2f sec." % (itot, time.time() - start_time))
                gc.collect()  

    # Repeat for second channel
    if nchannels > 1:
        itot = 0
        for ik, file in enumerate(files_Ch2):
            ip = iplanes[ik]
            if n_pages==1:
                with TiffReader(file) as tif:
                    im = tif.data() if HAS_SCANIMAGE else tif.pages[0].asarray()
                if im.dtype.type == np.uint16:
                    im = (im // 2)
                im = im.astype(np.int16)
                databases[ip]["meanImg_chan2"] += im.astype(np.float32)
                reg_file_chan2[ip].write(bytearray(im))
            else:
                tif, Ltif = open_tiff(file, not HAS_SCANIMAGE)
                ix = 0
                while 1:
                    im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
                    if im is None:
                        break
                    nframes = im.shape[0]
                    ix += nframes
                    itot += nframes
                    databases[ip]["meanImg_chan2"] += im.astype(np.float32).sum(axis=0)
                    reg_file_chan2[ip].write(bytearray(im))
                    if itot % 1000 == 0:
                        logger.info("%d frames of binary, time %0.2f sec." % (itot, time.time() - start_time))
                    gc.collect()

    # Update the databases with image dimensions and mean image
    for db in databases:
        db["Ly"], db["Lx"] = shape
        db["meanImg_chan1"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        # Save db and settings to each plane folder
        np.save(db["save_path0"], db)
        np.save(db["save_path0"], settings)

    return databases
  




    
