from qtpy.QtGui import QAction

from . import styles
from . import io


def fileMenu(parent):
    """Main Menu bar for the GUI"""

    ######## MAIN MENU BAR ########
    # Load database
    parent.load_db_btn = QAction("&Open DB File", parent)
    parent.load_db_btn.setShortcut("Ctrl+D")
    parent.load_db_btn.triggered.connect(lambda: io.load_db(parent, filename=None))

    # Load settings
    parent.load_settings_btn = QAction("&Open Settings File", parent)
    parent.load_settings_btn.setShortcut("Ctrl+O")
    parent.load_settings_btn.triggered.connect(
        lambda: io.load_settings(parent, filename=None)
    )

    # Revert to default settings
    parent.default_settings_btn = QAction("&Revert to default settings", parent)
    parent.default_settings_btn.setShortcut("Ctrl+R")
    parent.default_settings_btn.triggered.connect(
        lambda: io.revert_default_settings(parent)
    )

    # Save settings
    parent.save_settings_btn = QAction("&Save Settings File", parent)
    parent.save_settings_btn.setShortcut("Ctrl+S")
    parent.save_settings_btn.triggered.connect(lambda: io.save_settings(parent))

    # Save database
    parent.save_database_btn = QAction("&Save Database File", parent)
    parent.save_database_btn.setShortcut("Ctrl+B")
    parent.save_database_btn.triggered.connect(lambda: io.save_db(parent))

    # Exit GUI
    parent.exit_gui_btn = QAction("&Exit", parent)
    parent.exit_gui_btn.setShortcut("Ctrl+ESC")
    parent.exit_gui_btn.triggered.connect(lambda: parent.close())

    # Make main menu bar
    main_menu = parent.menuBar()
    main_menu.setStyleSheet(styles.menuStyle())
    main_menu.setNativeMenuBar(False)
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(parent.load_db_btn)
    file_menu.addAction(parent.load_settings_btn)
    file_menu.addAction(parent.default_settings_btn)
    file_menu.addAction(parent.save_settings_btn)
    file_menu.addAction(parent.save_database_btn)
    file_menu.addAction(parent.exit_gui_btn)


def batchMenu(parent):
    """Batch processing menu bar"""

    parent.add_batch_btn = QAction("&Add to Batch", parent)
    parent.add_batch_btn.setShortcut("Ctrl+A")
    parent.add_batch_btn.triggered.connect(lambda: parent.add_batch())

    parent.view_batch_btn = QAction("&View/Edit Batch List", parent)
    parent.view_batch_btn.setShortcut("Ctrl+V")
    parent.view_batch_btn.triggered.connect(lambda: parent.view_batch())

    main_menu = parent.menuBar()
    main_menu.setStyleSheet(styles.menuStyle())
    main_menu.setNativeMenuBar(False)
    batch_menu = main_menu.addMenu("&Batch")
    batch_menu.addAction(parent.add_batch_btn)
    batch_menu.addAction(parent.view_batch_btn)
