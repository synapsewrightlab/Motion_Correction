import pathlib
import os
import json

import numpy as np

from qtpy.QtWidgets import QFileDialog

from .. import parameters
from .gui_utils import COMBO_KEYS, FILE_KEYS, list_to_str, get_settings, get_db


def load_settings(parent, filename=None):
    """Function to get and load settings file"""
    # Get the settings file name
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Open settings file (npy or json)")
        filename = name[0]

    # Get the settings values
    if not filename:
        print("No File Selected")
        return

    else:
        if isinstance(filename, pathlib.Path):
            filename = str(filename)
        ext = os.path.splitext(filename)[1]
        if ext == ".json":
            with open(filename, "r") as f:
                settings_in = json.load(f)
        else:
            settings_in = np.load(filename, allow_pickle=True).item()

    # Reset the Gui settings to the loaded values
    set_default_settings(parent.SETTINGS, settings_in)
    set_settings(parent.SETTINGS, parent.settings_gui)


def load_db(parent, filename=None):
    """Function to get and load database file"""
    # Get the database file name
    if filename is None:
        name = QFileDialog.getOpenFileName(parent, "Open database file (npy or json)")
        filename = name[0]

    if not filename:
        print("No File Selected")
        return

    else:
        if isinstance(filename, pathlib.Path):
            filename = str(filename)
        ext = os.path.splitext(filename)[1]
        if ext == ".json":
            with open(filename, "r") as f:
                db = json.load(f)
        else:
            db = np.load(filename, allow_pickle=True).item()

    # Reset the gui db values using the loaded db
    set_default_db(parent.DB, db)
    set_db(parent.DB, parent.db_gui, parent)


def get_file_folders(parent, ftype="data_path", name=None):
    """Function to get file folder paths"""
    if name is None:
        name = QFileDialog.getExistingDirectory(parent, "choose file folder")

    if len(name) > 0:
        print(f"adding {name} to {ftype}")
        if ftype == "data_path":
            for i in range(9):
                if len(parent.db_gui["data_path"][i].text()) == 0:
                    parent.db_gui["data_path"][i].setText(name)
                    parent.db_gui["data_path"][i].setToolTip(name)
                    parent.data_path_checkboxes[i].setVisible(True)
                    break
            # Update buttons
            parent.runButton.setEnabled(True)
            # parent.addToBatch.setEnabled(True)
            parent.remove_path_btn.setEnabled(True)
        else:
            parent.db_gui[ftype].setText(name)
            parent.db_gui[ftype].setToolTip(name)


def remove_selected_paths(parent):
    """Function to remove selected paths"""
    remaining_paths = []
    for i in range(9):
        if not parent.data_path_checkboxes[i].isChecked():
            path_text = parent.db_gui["data_path"][i].text()
            if len(path_text) > 0:
                remaining_paths.append(path_text)
        else:
            path_text = parent.db_gui["data_path"][i].text()
            if len(path_text) > 0:
                print(f"removing {path_text} from data_path")

    # Clear all fields, checkboxed and hide them
    for i in range(9):
        parent.db_gui["data_path"][i].setText("")
        parent.db_gui["data_path"][i].setToolTip("")
        parent.data_path_checkboxes[i].setChecked(False)
        parent.data_path_checkboxes[i].setVisible(False)

    # Repopulate the remaining paths
    for i, path in enumerate(remaining_paths):
        parent.db_gui["data_path"][i].setText(path)
        parent.db_gui["data_path"][i].setTooTip(path)
        parent.data_path_checkboxes[i].setVisible(True)

    # Disable buttons is no datapaths remain
    if len(remaining_paths) == 0:
        parent.runButton.setEnabled(False)
        # parent.addToBatch.setEnabled(False)
        parent.remove_path_btn.setEnabled(False)


def save_settings(parent):
    """Function to save current settings to file"""
    name = QFileDialog.getSaveFileName(parent, "Settings name (*.npy)")
    settings = get_settings(parent.SETTINGS, parent.settings_gui)
    if len(name) > 0 and name[0] is not None:
        name = name[0]
        np.save(name, settings)
        print(f"saved current settings to {name}")


def save_db(parent):
    """FUnction to save the current db to file"""
    name = QFileDialog.getSaveFileName(parent, "Database name (*.npy)")
    db = get_db(parent.DB, parent.db_gui)
    if len(name) > 0 and name[0] is not None:
        name = name[0]
        np.save(name, db)
        print(f"saved current database to {name}")


def save_batch_db_settings(parent):
    """Helper function to save db and batches for each batch"""

    print("saving db and settings")
    db = get_db(parent.DB, parent.db_gui)
    settings = get_settings(parent.SETTINGS, parent.settings_gui)
    save_path0 = (
        db["save_path0"] if db["save_path0"] is not None else db["data_path"][0]
    )
    parent.save_path = os.path.join(save_path0, db["save_folder"])
    settings_file = os.path.join(parent.save_path, "settings.npy")
    db_file = os.path.join(parent.save_path, "db.npy")
    if not os.path.isdir(parent.save_path):
        os.makedirs(parent.save_path)
    np.save(settings_file, settings)
    np.save(db_file, db)


def set_default_db(DB, db):
    """Helper function to reset gui DB"""
    for key in DB.keys():
        DB[key]["default"] = db[key]


def set_db(DB, db_gui, parent=None):
    """Helper function to set the gui DB"""
    for key in db_gui.keys():

        if DB[key]["type"] == bool:
            db_gui[key].setChecked(DB[key]["default"])

        elif key in FILE_KEYS:
            if key == "data_path":
                for i in range(9):
                    if i < len(DB[key]["default"]):
                        db_gui[key][i].setText(DB[key]["default"][i])
                        if parent is not None:
                            parent.data_path_checkboxes[i].setVisible(True)
                    else:
                        db_gui[key][i].setText("")
                        if parent is not None:
                            parent.data_path_checkboxes[i].setVisible(False)
                # Enable/disable remove button based on whether paths exist
                if parent is not None and len(DB[key]["default"]) > 0:
                    parent.remove_path_btn.setEnabled(True)
            else:
                db_gui[key].setText(DB[key]["default"])
        else:
            if DB[key]["default"] is not None:
                if key in COMBO_KEYS:
                    all_items = [
                        db_gui[key].itemText(i) for i in range(db_gui[key].count())
                    ]
                    db_gui[key].setCurrentIndex(all_items.index(DB[key]["default"]))
                elif DB[key]["type"] == list or DB[key]["type"] == tuple:
                    db_gui[key].setText(list_to_str(DB[key]["default"]))
                else:
                    db_gui[key].setText(str(DB[key]["default"]))
            else:
                db_gui[key].setText("")


def set_default_settings(SETTINGS, settings):
    """Helper function to reset the gui SETTINGS"""
    for key in SETTINGS.keys():
        if "gui_name" not in SETTINGS[key]:
            set_default_settings(SETTINGS[key], settings[key])
        else:
            SETTINGS[key]["default"] = settings[key]


def set_settings(SETTINGS, settings_gui):
    """Helper function to set the gui settings"""
    for key in settings_gui.keys():
        if "gui_name" not in SETTINGS[key]:
            set_settings(SETTINGS[key], settings_gui[key])

        elif SETTINGS[key]["type"] == bool:
            settings_gui[key].setChecked(SETTINGS[key]["default"])

        else:
            if SETTINGS[key]["default"] is not None:
                if key in COMBO_KEYS:
                    all_items = [
                        settings_gui[key].itemText(i)
                        for i in range(settings_gui[key].count())
                    ]
                    settings_gui[key].setCurrentIndex(
                        all_items.index(SETTINGS[key]["default"])
                    )

                elif SETTINGS[key]["type"] == list or SETTINGS[key]["type"] == tuple:
                    settings_gui[key].setText(list_to_str(SETTINGS[key]["default"]))

                else:
                    settings_gui[key].setText(str(SETTINGS[key]["default"]))

            else:
                settings_gui[key].setText("")


def revert_default_settings(parent):
    """Function to revert to default settings"""
    # Reset the Gui settings to the loaded values
    settings_in = parameters.SETTINGS
    set_default_settings(parent.SETTINGS, settings_in)
    set_settings(parent.SETTINGS, parent.settings_gui)
