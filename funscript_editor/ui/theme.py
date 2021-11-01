
from PyQt5 import QtGui, QtCore ,QtWidgets

from funscript_editor.utils.config import SETTINGS
import funscript_editor.ui.breeze_resources


def setup_theme():
        try:
            if SETTINGS['dark_theme']:
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    file = QtCore.QFile(":/dark/stylesheet.qss")
                    file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
                    stream = QtCore.QTextStream(file)
                    app.setStyleSheet(stream.readAll())
        except: pass
