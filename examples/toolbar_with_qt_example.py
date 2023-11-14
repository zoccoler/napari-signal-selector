from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QToolBar, QToolButton
from qtpy.QtGui import QIcon
from pathlib import Path

icon_root = Path(__file__).parent.parent / "src/napari_signal_selector/icons"

# class MyMainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         layout = QVBoxLayout(self.central_widget)

#         # Custom toolbar
#         self.toolbar = QToolBar()
#         self.add_custom_buttons(self.toolbar)
#         layout.addWidget(self.toolbar)

#     def add_custom_buttons(self, toolbar):
#         # Paths to icons
#         print(Path(icon_root / "select.png"))
#         default_icon_path = Path(icon_root / "select.png").__str__()
#         checked_icon_path = Path(icon_root / "span_select.png").__str__()

#         # Load icons
#         default_icon = QIcon(default_icon_path)
#         checked_icon = QIcon(checked_icon_path)  # Not directly used, but here for consistency

#         # Create the button
#         button = QToolButton()
#         button.setCheckable(True)
#         button.setIcon(default_icon)

#         # Set stylesheet for checked state with variable path
#         button.setStyleSheet(f"""
#             QToolButton:checked {{
#                 image: url({checked_icon_path});
#             }}
#         """)

#         # Connect the button's toggled signal to a slot
#         button.toggled.connect(self.button_toggled)

#         toolbar.addWidget(button)

#     def button_toggled(self, checked):
#         print("Button checked:", checked)

# # Usage
# app = QApplication([])
# window = MyMainWindow()
# window.show()
# app.exec_()

from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QToolBar, QToolButton
from qtpy.QtGui import QIcon

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Custom toolbar
        self.toolbar = QToolBar()
        self.add_custom_buttons(self.toolbar)
        layout.addWidget(self.toolbar)

    def add_custom_buttons(self, toolbar):
        # Paths to icons
        self.default_icon_path = Path(icon_root / "select.png").__str__()
        self.checked_icon_path = Path(icon_root / "select_checked.png").__str__()

        # Load icons
        self.default_icon = QIcon(self.default_icon_path)
        self.checked_icon = QIcon(self.checked_icon_path)

        # Create the button
        button = QToolButton()
        button.setCheckable(True)
        button.setIcon(self.default_icon)

        # Connect the button's toggled signal to a slot
        button.toggled.connect(self.handle_button_toggle)

        toolbar.addWidget(button)

    def handle_button_toggle(self, checked):
        # Get the button that sent the signal
        button = self.sender()

        # Change the icon based on the checked state
        if checked:
            button.setIcon(self.checked_icon)
        else:
            button.setIcon(self.default_icon)

# Usage
app = QApplication([])
window = MyMainWindow()
window.show()
app.exec_()


