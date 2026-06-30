from qtpy.QtGui import QAction

from . import styles

def fileMenu(parent):
    """Main Menu bar for the GUI"""

    ######## MAIN MENU BAR ########
    # Load database
    parent.load_db = QAction("&Open DB File", parent)
    parent.load_db.setShortcut("Ctrl+D")
    parent.load_db.triggered.connect(lambda: print("Add function"))

    # Load settings
    parent.load_settings = QAction("&Open Settings File", parent)
    parent.load_settings.setShortcut("Ctrl+O")
    parent.load_settings.triggered.connect(lambda: print("Add function"))

    # Revert to default settings
    parent.default_settings = QAction("&Revert to default settings")
    parent.default_settings.setShortcut("Ctrl+R")
    parent.default_settings.triggered.connect(lambda: print("Add function"))

    # Save settings
    parent.save_settings = QAction("&Save Settings File", parent)
    parent.save_settings.setShortcut("Ctrl+S")
    parent.save_settings.triggered.connect(lambda: print("Add function"))

    # Exit GUI
    parent.exit_gui = QAction("&Exit", parent)
    parent.exit_gui.setShortcut("Ctrl+ESC")
    parent.exit_gui.triggered.connect(lambda: parent.close())

    # Make main menu bar
    main_menu = parent.menuBar()
    main_menu.setStyleSheet(styles.menuStyle())
    main_menu.setNativeMenuBar(False)
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(parent.load_db)
    file_menu.addAction(parent.load_settings)
    file_menu.addAction(parent.default_settings)
    file_menu.addAction(parent.save_settings)
    file_menu.addAction(parent.exit_gui)


def batchMenu(parent):
    """Batch processing menu bar"""

    parent.add_batch = QAction("&Add to Batch")
    parent.add_batch.setShortcut("Ctrl+A")
    parent.add_batch.triggered.connect(lambda: print("Add function"))

    parent.view_batch = QAction("&View/Edit Batch List")
    parent.view_batch.setShortcut("Ctrl+V")
    parent.view_batch.triggered.connect(lambda: print("Add function"))

    main_menu = parent.menuBar()
    main_menu.setStyleSheet(styles.menuStyle())
    main_menu.setNativeMenuBar(False)
    batch_menu = main_menu.addMenu("&Batch")
    batch_menu.addAction(parent.add_batch)
    batch_menu.addAction(parent.view_batch)
