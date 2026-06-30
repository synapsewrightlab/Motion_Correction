from pathlib import Path
from qtpy import QtCore, QtGui, QtWidgets
import sys, os
import logging
import traceback
import io

import numpy as np
import torch

from .. import logger, run_motion_correct
from . import styles

DB_KEYS = [  "input_format", "look_one_level_down",
             "keep_movie_raw", "nplanes", "nchannels", "swap_order",
             "functional_chan", "ignore_flyback", "save_folder",
             "batch_size",
             "h5py_key", "nwb_series", "force_sktiff"
        ]
FILE_KEYS = ["data_path", "save_path0", "fast_disk"]
COMBO_KEYS = ["input_format", "algorithm"]

class MyLog(QtCore.QObject):
    """
    Create a new signal. Needs to be static element. Inherits from QObject to emit signals
    """
    signal = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

class ThreadLogger(logging.Handler):
    """
    Custom logging handler that can run in seperate thread and emit all logs 
    via signals/slots so they can be used to update the GUI in the main thread
    """
    def __init__(self):
        super().__init__()
        self.log = MyLog()

    # logging.Handler.emit() is intended to be implemented by subclasses
    def emit(self, record):
        msg = self.format(record)
        self.log.signal.emit(msg)

class MotionCorrectWorker2(QtCore.QThread):
    """
    Class to handle working in other threads
    """
    finished = QtCore.Signal(str)

    def __init__(self, parent, db_file, settings_file):
        super(MotionCorrectWorker, self).__init__()
        self.db_file = db_file
        self.settings_file = settings_file
        self.parent = parent
        self.logHandler = ThreadLogger()

    def run(self):
        db = np.load(self.db_file, allow_pickle=True).item()
        settings = np.load(self.settings_file, allow_pickle=True).item()
        try:
            logger_setup(db['save_path'])
            run_motion_correct(db=db, settings=settings)
            self.finished.emit("finished")

        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()
            self.finished.emit("error")


class MotionCorrectWorker(QtCore.QObject):
    """
    Worker that runs motion correction in a separate process to avoid 
    QThread stack limitations on macOS.
    """
    def __init__(self, parent, db_file, settings_file):
        super(MotionCorrectWorker, self).__init__()
        self.db_file = db_file
        self.settings_file = settings_file
        self.parent = parent
        self.process = None

    def start(self):
        """Start motion correction in a separate process using QProcess."""
        self.process = QtCore.QProcess()
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)

        # Create a Python script to run suite2p
        script = f'''
import numpy as np
from Motion_Correction.run_motion_correct import logger_setup, run_motion_correction, get_save_folder

db = np.load("{self.db_file}", allow_pickle=True).item()
settings = np.load("{self.settings_file}", allow_pickle=True).item()

logger_setup(get_save_folder(db))
run_motion_correction(db=db, settings=settings)
'''
        self.process.start(sys.executable, ["-c", script])

    def _on_output(self):
        """Handle output from the subprocess."""
        if self.process:
            data = self.process.readAllStandardOutput()
            text = bytes(data).decode("utf-8", errors="replace")
            print(text, end="")

    def _on_finished(self, exit_code, exit_status):
        """Handle process completion."""
        if exit_code == 0:
            self.finished.emit("finished")
        else:
            self.finished.emit("error")

    def terminate(self):
        """Terminate the subprocess if running."""
        if self.process and self.process.state() != QtCore.QProcess.NotRunning:
            self.process.terminate()

    def quit(self):
        """Stop the subprocess (alias for terminate, for QThread compatibility)."""
        self.terminate()

    def wait(self):
        """Wait for the process to finish (for compatibility)."""
        if self.process:
            self.process.waitForFinished(-1)

    def isRunning(self):
        """Check if the process is still running (for QThread compatibility)."""
        if self.process:
            return self.process.state() == QtCore.QProcess.Running
        return False


class QtHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        record = self.format(record)
        if record:
            XStream.stdout().write(f"{record}")


class XStream(QtCore.QObject):
    _stdout = None
    _stderr = None
    messageWritten = QtCore.Signal(str)

    def flush(self):
        pass

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

    def write(self, msg):
        if not self.signalsBlocked():
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        if not XStream._stdout:
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if not XStream._stderr:
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr


def logger_setup(name):
    """Sets up logger"""
    logger = logging.getLogger(name)
    handler = QtHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger

def list_to_str(l):
    return ", ".join(str(l0) for l0 in l)


def create_input(key, SETTINGS, settings_gui, width=160):
    """Function to help create inputs"""

    qlabel = QtWidgets.QLabel(SETTINGS[key]["gui_name"])
    qlabel.setToolTip(SETTINGS[key]["description"])
    qlabel.setFixedWidth(width)
    qlabel.setAlignment(QtCore.Qt.AlignRight)
    qlabel.setStyleSheet("QLabel {font-family: Arial; font-weight:bold; color: white}")

    if SETTINGS[key]["type"] == bool:
        settings_gui[key] = QtWidgets.QCheckBox()
        settings_gui[key].setChecked(SETTINGS[key]["default"])

    else:
        settings_gui[key] = QtWidgets.QLineEdit()
        settings_gui[key].setFixedWidth(100)
        
        if SETTINGS[key]["default"] is not None:
            if key in COMBO_KEYS:
                settings_gui[key] = QtWidgets.QComboBox()
                strs = SETTINGS[key]["description"].split("[")[1].split("]")[0].split(", ")
                strs = [s[1:-1] for s in strs]
                settings_gui[key].addItems(strs)
                settings_gui[key].setFixedWidth(100)
                settings_gui[key].setStyleSheet(styles.comboBoxStyle())
            elif SETTINGS[key]["type"] == list or SETTINGS[key]["type"] == tuple:
                settings_gui[key].setText(list_to_str(SETTINGS[key]["default"]))
            else:
                settings_gui[key].setText(str(SETTINGS[key]["default"]))
    
    settings_gui[key].setToolTip(SETTINGS[key]["description"])

    return qlabel
