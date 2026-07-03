from qtpy.QtWidgets import (
    QDialog,
    QWidget,
    QGridLayout,
    QLabel,
    QListWidget,
    QAbstractItemView,
    QPushButton,
)


class BatchView(QDialog):
    """Custom QDialog for editable batch list"""

    def __init__(self, parent=None):
        super(BatchView, self).__init__(parent)
        # self.setGeometry(300, 300, 600, 320)
        self.setWindowTitle("Batch List")

        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        layout.addWidget(QLabel("(Select multiple using ctrl)"), 0, 0, 1, 1)

        self.list = QListWidget(parent)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.list, 1, 0, 10, 1)
        self.lsit.addItems(parent.batch_list)

        remove = QPushButton("Remove selection")
        remove.clicked.connect(lambda: self.remove_selected(parent))
        layout.addWidget(remove, 11, 0, 1, 1)

        done = QPushButton("Close")
        done.clicked.connect(self.exit_list)
        layout.addWidget(done, 11, 1, 1, 1)

    def remove_selected(self, parent):
        for item in self.list.selectedItems():
            parent.batch_list.removeItem(item.text())
            print("Removed", item.text())
        parent.list.clear()
        parent.list.addItems(parent.batch_list)

    def exit_list(self):
        self.accept()
