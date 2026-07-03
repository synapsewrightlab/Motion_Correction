import sys
import os
import warnings

from qtpy import QtGui, QtCore
from qtpy.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QPlainTextEdit,
    QScrollArea,
)

from superqt import QCollapsible

from .batch_window import BatchView
from . import styles, menus, gui_utils, io
from .. import parameters

import logging

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main window for the motion correction GUI
    """

    def __init__(self):
        super(MainWindow, self).__init__()

        # Setup some basic window properties
        screen_size = QtGui.QGuiApplication.primaryScreen().geometry()
        win_h = int(screen_size.height() * 0.9)
        win_w = int(1500)
        self.setGeometry(50, 50, win_w, win_h)
        self.setStyleSheet("background-color: black")
        self.setWindowTitle("Run Motion Correction (select settings and parameters)")

        self.initGUI()

    def initGUI(self):
        """Create the GUI window"""
        # Initialize attributes and parameters
        self.DB = parameters.DATABASE
        self.SETTINGS = parameters.SETTINGS
        parameters.add_descriptions(self.DB, dstr="db")
        parameters.add_descriptions(self.SETTINGS, dstr="settings")

        self.data_path = []
        self.save_path = []
        self.fast_disk = []
        self.ibatch = 0
        self.batch_list = []

        # Set up the central widget window
        self.cwidget = QWidget(self)
        self.layout = QGridLayout()
        self.cwidget.setLayout(self.layout)
        self.setCentralWidget(self.cwidget)

        self.setPalette(styles.DarkPalette())

        # Great main menu bar
        menus.fileMenu(self)
        menus.batchMenu(self)

        # Add database and settings inputs
        self.create_db_settings_inputs()

    def create_db_settings_inputs(self):  # Should really break this up
        """Method to populate the widget with db and settings inputs"""
        self.db_gui = {}
        self.settings_gui = {}

        XLfont = (
            "QLabel {font-size: 12pt; font: Arial; font-weight: bold; color: white}"
        )
        bigfont = (
            "QLabel {font-size: 10pt; font: Arial; font-weight: bold; color: white}"
        )

        qlabel = QLabel("File paths")
        qlabel.setStyleSheet(XLfont)
        self.layout.addWidget(qlabel, 0, 0, 1, 1)

        # data path
        for b, key in enumerate(["input_format", "one_down"]):
            qlabel = gui_utils.create_input(
                key, self.DB, self.db_gui, width=80 + b * 50
            )
            self.layout.addWidget(qlabel, 1, 2 * b, 1, 1)
            self.layout.addWidget(self.db_gui[key], 1, 2 * b + 1, 1, 1)

        cw = 5
        b = 2
        self.path_btns = []
        kinfos = ["folders", "(default is 1st data_path)", "(default is save_path0)"]
        FILE_KEYS = ["data_path", "save_path0", "fast_disk"]
        for kinfo, key in zip(kinfos, FILE_KEYS):
            btn = QPushButton(f"add {key} {kinfo}")
            btn.setStyleSheet(styles.pushButtonStyle())
            btn.clicked.connect(
                lambda state, x=key: io.get_file_folders(self, ftype=x, name=None)
            )
            btn.setToolTip(self.DB["data_path"]["description"])
            self.path_btns.append(btn)
            self.layout.addWidget(btn, b, 0, 1, cw)
            b += 1
            if key != "data_path":
                self.db_gui[key] = QLabel("")
                self.layout.addWidget(self.db_gui[key], b, 0, 1, cw)
                b += 1
            else:
                self.db_gui[key] = [QLabel("") for _ in range(9)]
                for label in self.db_gui[key]:
                    label.setStyleSheet("QLabel {color: white}")
                self.data_path_checkboxes = [QCheckBox() for _ in range(9)]
                for i in range(9):
                    self.layout.addWidget(self.data_path_checkboxes[i], b, 0, 1, 1)
                    self.layout.addWidget(self.db_gui[key][i], b, 1, 1, cw - 1)
                    self.data_path_checkboxes[i].setVisible(False)  # Initially hidden
                    b += 1

                # Add remove button for selected data paths
                self.remove_path_btn = QPushButton("remove selected data paths")
                self.remove_path_btn.clicked.connect(
                    lambda: io.remove_selected_paths(self)
                )
                self.remove_path_btn.setToolTip(
                    "Remove checked data_path from the list"
                )
                self.remove_path_btn.setStyleSheet(styles.pushButtonStyle())
                self.remove_path_btn.setEnabled(False)  # Initially disabled
                self.layout.addWidget(self.remove_path_btn, b, 0, 1, cw)
                b += 1

        n0 = 17

        # Add run button
        self.runButton = QPushButton("Run Motion Correction")
        self.runButton.clicked.connect(lambda: self.run_motion_correction())
        self.runButton.setEnabled(False)
        self.runButton.setStyleSheet(styles.pushButtonStyle())
        self.layout.addWidget(self.runButton, n0, 0, 1, 1)
        # Add stop button
        self.stopButton = QPushButton("STOP")
        self.stopButton.clicked.connect(lambda: self.stop())
        self.stopButton.setEnabled(False)
        self.stopButton.setStyleSheet(styles.pushButtonStyle())
        self.layout.addWidget(self.stopButton, n0, 1, 1, 1)

        # Log stream
        self.textEdit = QPlainTextEdit(self)
        self.textEdit.setStyleSheet(styles.textEditStyle())
        self.layout.addWidget(self.textEdit, n0 + 1, 0, 16, cw + 2)

        gui_utils.XStream.stdout().messageWritten.connect(self.update_text)
        gui_utils.XStream.stderr().messageWritten.connect(self.update_text)

        # File settings
        fileSettingLabel = QLabel("File settings")
        fileSettingLabel.setStyleSheet(XLfont)
        fileSettingLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(fileSettingLabel, 0, cw, 1, 2)
        b = 1
        for key in gui_utils.DB_KEYS[2:]:
            temp_label = gui_utils.create_input(key, self.DB, self.db_gui)
            self.layout.addWidget(temp_label, b, cw, 1, 1)
            self.layout.addWidget(self.db_gui[key], b, cw + 1, 1, 1)
            b += 1

        # General settings
        self.general_keys = [
            "algorithm",
            "torch_device",
            "save_tif",
            "downsampled_frames",
        ]
        genSettingLabel = QLabel("General settings")
        genSettingLabel.setAlignment(QtCore.Qt.AlignCenter)
        genSettingLabel.setStyleSheet(XLfont)
        self.layout.addWidget(genSettingLabel, b + 1, cw, 1, 2)
        b += 2
        for key in self.general_keys:
            temp_label = gui_utils.create_input(key, self.SETTINGS, self.settings_gui)
            self.layout.addWidget(temp_label, b, cw, 1, 1)
            self.layout.addWidget(self.settings_gui[key], b, cw + 1, 1, 1)
            b += 1

        # Scroll area
        scrollArea = QScrollArea()
        scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scrollArea.setStyleSheet(styles.scrollStyle())
        scrollArea.setWidgetResizable(True)
        swidget = QWidget(self)
        scrollArea.setWidget(swidget)
        layoutScroll = QGridLayout()
        swidget.setLayout(layoutScroll)
        self.layout.addWidget(scrollArea, 0, cw + 2, n0 + 17, 4)

        # Add alogrithm specific settings
        drop_labels = ["Suite2p", "Patchwarp"]
        b = 0
        for dlabel in drop_labels:
            qbox = QCollapsible(f"{dlabel} settings")
            dlabel = dlabel + "_settings"
            qbox._toggle_btn.setStyleSheet(styles.collButtonStyle())
            qboxG = QGridLayout()
            _content = QWidget()
            _content.setLayout(qboxG)
            bl = 0
            self.settings_gui[dlabel] = {}
            for key in self.SETTINGS[dlabel].keys():
                if "gui_name" in self.SETTINGS[dlabel][key]:
                    temp_label = gui_utils.create_input(
                        key, self.SETTINGS[dlabel], self.settings_gui[dlabel]
                    )
                    qboxG.addWidget(temp_label, bl, 0, 1, 1)
                    qboxG.addWidget(self.settings_gui[dlabel][key], bl, 1, 1, 1)
                    bl += 1

            qbox.setContent(_content)

            if dlabel == "Suite2p settings":
                qbox.expand(animate=False)
            else:
                qbox.collapse(animate=False)

            layoutScroll.addWidget(qbox, b, 0, 1, 1)
            b += 1

        layoutScroll.addWidget(QLabel(""), b, 0, 1, 1)
        layoutScroll.setColumnStretch(0, 0)

    def add_batch(self):
        print(f"Adding {self.save_path} to batch")
        io.save_batch_db_settings(self)
        self.batch_list.append(self.save_path)
        L = len(self.batch_list)

        # Clear db
        io.set_default_db(self.DB, parameters.default_db())
        io.set_db(self.DB, self.db_gui, parent=self)

        # And enable the run button
        self.runButton.setEnabled(True)

    def view_batch(parent):
        BV = BatchView(parent)
        result = BV.exec_()

    def run_motion_correction(self):
        self.finish = True
        self.error = False

        if len(self.batch_list) == 0:
            io.save_batch_db_settings(self)
        else:
            self.save_path = self.batch_list[self.ibatch]

        settings_file = os.path.join(self.save_path, "settings.npy")
        db_file = os.path.join(self.save_path, "db.py")

        print("Running Motion Correction with command:")
        cmd = f"-m motion_correction -- settings {settings_file} -- db {db_file} --verbose"
        print("python" + cmd)

        self.worker = gui_utils.MotionCorrectWorker(
            self, db_file=db_file, settings_file=settings_file
        )
        self.worker.finished.connect(self.finished)

        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)

        self.worker.start()

    # define a new slot the recives a string
    @QtCore.Slot(str)
    def update_text(self, log_text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(log_text)
        self.textEdit.ensureCursorVisible()

    def stop(self):
        self.finish = False
        self.worker.quit()
        self.worker.terminate()
        self.finished(msg=None)

    @QtCore.Slot(str)
    def finished(self, msg=None):
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        print(msg)

        if msg is None:
            self.textEdit.appendPlainText("Interrupted by user (not finished)\n")

        elif msg == "finished":
            if len(self.batch_list) == 0:
                pass
            else:
                self.textEdit.appendPlainText(
                    "BATCH MODE %d more recordings remaining \n"
                    % (len(self.batch_list) - self.ibatch - 1)
                )
                self.ibatch += 1
                if self.ibatch < len(self.batch_list):
                    self.run_motion_correction()

        elif msg == "error":
            self.textEdit.appendPlainText("Interrupted by error (not finished)\n")

    def closeEvent(self, event):
        """Custom close window function"""

        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
