import sys
import os
import warnings

from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QHBoxLayout, QLabel

from . import styles, menus, gui_utils
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
        win_w = int(screen_size.width() * 0.9)
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


    def create_db_settings_inputs(self):
        """Method to populate the widget with db and settings inputs"""
        self.db_gui = {}
        self.settings_gui = {}

        XLfont = "QLabel {font-size: 12pt; font: Arial; font-weight: bold; color: white}"
        bigfont = "QLabel {font-size: 10pt; font: Arial; font-weight: bold; color: white}"

        qlabel = QLabel("File paths")
        qlabel.setStyleSheet(XLfont)
        self.layout.addWidget(qlabel, 0, 0, 1, 1)

        # data path
        qlabel = gui_utils.create_input("input_format", self.DB, self.db_gui, width=80 + 1*50)
        self.layout.addWidget(qlabel, 1, 2*0, 1, 1)
        self.layout.addWidget(self.db_gui["input_format"], 1, 2*0+1, 1, 1)
        



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


