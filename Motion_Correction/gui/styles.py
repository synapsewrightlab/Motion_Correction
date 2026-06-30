from qtpy import QtGui
from qtpy.QtGui import QFont

def menuStyle():
    menu_style = """
    QMenuBar {
        background-color: rgb(50,50,50);
        color:white;
        }
    QMenuBar::item {
        background-color: rgb(50,50,50);
        color: white;
        }
    QMenuBar::item:selected {
        background-color: rgb(212,212,212);
        color: black;
        }
    QMenu {
        background-color: rgb(50,50,50);
        color: white;
        }
    QMenu::item {
        background-color: rgb(50,50,50);
        color: white;
        }
    QMenu::item:selected {
        background-color: rgb(212,212,212);
        color: black;
        }

"""
    return menu_style


def checkBoxStyle():
    check_style = """
    QCheckBox {
        color:white;
    
    }

"""
    return check_style


def comboBoxStyle():
    combo_style = """
        QComboBox {
        border: 1px solid rgb(100,100,100);
        border-radius: 2px;
        padding: 0px 6px;
        min-width: 8em;
        background-color: rgb(20,20,20);
        color: white;
        font-family: Arial;
        }

        /* ComboBox when hovered */
        QComboBox:hover {
            border: 2px solid rgb(42,130,218);
        }

        /* ComboBox when open/focused */
        QComboBox:on {
            border-bottom-left-radius: 0px;
            border-bottom-right-radius: 0px;
        }

        /* Dropdown Arrow Container Sub-control */
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: bottom right;
            width: 20px;
            border-left-width: 2px; /* Removes native divider */
            background: rgb(40,40,40)
        }

        /* The Popup List View Container */
        QComboBox QAbstractItemView {
            border: 2px solid rgb(50,50,50);
            border-top-width: 0px; /* Avoid double borders with main box */
            border-bottom-left-radius: 6px;
            border-bottom-right-radius: 6px;
            background-color: rgb(50,50,50);
            color: white;
            outline: 0px; /* Removes focus dotted line */
        }

        /* Individual items in the dropdown list */
        QComboBox QAbstractItemView::item {
            min-height: 20px;
            padding-left: 5px;
        }

        /* Hovered or Selected item in the list */
        QComboBox QAbstractItemView::item:selected {
            background-color: rgb(42,130,218);
            color: black;
        }
        QComboBox QAbstractItemView::item:hover {
            background-color: rgb(14,71,128);
            color: white;
        }

"""

    return combo_style


class DarkPalette(QtGui.QPalette):
    """Class that inherits from pyqtgraph.QtGui.QPalette and renders dark colours for the application.
    (from pykilosort/kilosort4)
    """

    def __init__(self):
        QtGui.QPalette.__init__(self)
        self.setup()

    def setup(self):
        self.setColor(QtGui.QPalette.Window, QtGui.QColor(50, 50, 50))
        self.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Base, QtGui.QColor(0, 0, 0))
        self.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(0, 0, 0))
        self.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Button, QtGui.QColor(40, 40, 40))
        self.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        self.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text,
                      QtGui.QColor(128, 128, 128))
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.ButtonText,
            QtGui.QColor(128, 128, 128),
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.WindowText,
            QtGui.QColor(128, 128, 128),
        )




