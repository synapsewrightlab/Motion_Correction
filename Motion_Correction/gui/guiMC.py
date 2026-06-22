import sys
import os
import warnings

from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QMainWindow, QApplication, QDesktopWidget

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
        screen_size = QDesktopWidget().screenGeometry()
        win_h = int(screen_size.height() * 0.9)
        win_w = int(screen_size.width() * 0.9)
        self.setGeometry(50, 50, win_w, win_h)
        self.setStyleSheet("background:black")
        self.setWindowTitle("Run Motion Correction (select settings and parameters)")

        self.initGUI()

    def initGUI(self):
        """Create the GUI window"""
        # Initialize attributes and parameters



def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


