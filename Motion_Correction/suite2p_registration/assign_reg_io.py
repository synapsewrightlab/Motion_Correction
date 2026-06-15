import os

def assign_reg_io(
    f_raw_chan1,
    f_reg_chan1,
    f_raw_chan2=None,
    f_reg_chan2=None,
    align_by_chan2=False,
    save_path=None,
    save_tif=False,
):
    """
    Assign input/output arrays and save directories for registration I/O

    Determins which channel is the alignment socure and which is alternate.
    Sets up save directories, as well as for tiff files

    PARAMETERS
        f_raw_chan1 : np.ndarray or BinaryFile
            Raw functional channel frames.

        f_reg_chan1 : np.ndarray or BinaryFile
            Registered functional channel frames.

        f_raw_chan2 : np.ndarray or BinaryFile
            Raw second channel frames.

        f_reg_chan2 : np.ndarray or BinaryFile or None
            Registered second channel frames.

        align_by_chan2 : bool
            If True, use the second channel as the alignment source.

        save_path : str or None
            Base directory for saving registered tiff files.

        save_tif : bool
            If True, save registered functional channel frames as tiffs.

    OUTPUT
        f_align_in : np.ndarray or BinaryFile
            Input frames for alignment.

        f_align_out : np.ndarray or BinaryFile or None
            Output destination for aligned frames.

        f_alt_in : np.ndarray or BinaryFile or None
            Input frames for the alternate channel.

        f_alt_out : np.ndarray or BinaryFile or None
            Output destination for shifted alternate channel frames.

        tif_root_align : str or None
            Tiff output directory for the alignment channel.

        tif_root_alt : str or None
            Tiff output directory for the alternate channel. 
    
    """

    if f_raw_chan2 is None or not align_by_chan2:
        f_align_in = f_raw_chan1
        f_alt_in = f_raw_chan2
        f_align_out = f_reg_chan1
        f_alt_out = f_reg_chan2
    else:
        f_align_in = f_raw_chan2
        f_alt_in = f_raw_chan1
        f_align_out = f_reg_chan2
        f_alt_out = f_reg_chan1

    if f_alt_in is not None:
        if f_align_in.shape[0] != f_alt_in.shape[0]:
            raise ValueError("Number of frames in f_align_in and f_alt_in must match")
    
    # Set up tiff file saves if specified
    tif_align_out, tif_alt_out = None, None

    if save_tif:
        tif_chan1_save = os.path.join(save_path, "Ch1", "tif_files")
        os.makedirs(tif_chan1_save, exist_ok=True)
        if not align_by_chan2:
            tif_align_out = tif_chan1_save
        else:
            tif_alt_out = tif_chan1_save

        if f_reg_chan2 is not None:
            tif_chan2_save = os.path.join(save_path, "Ch2", "tif_files")
            os.makedirs(tif_chan2_save, exist_ok=True)  
        else:
            tif_chan2_save = None

        if align_by_chan2:
            tif_align_out = tif_chan2_save
        else:
            tif_alt_out = tif_chan2_save

    return f_align_in, f_align_out, f_alt_in, f_alt_out, tif_align_out, tif_alt_out
        