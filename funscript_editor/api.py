import sys
import logging
import os
import cv2

from funscript_editor.utils.logging import setup_logging
from funscript_editor.ui.funscript_editor_window import FunscriptEditorWindow
from funscript_editor.ui.minimal import MinimalFunscriptGenerator

from PyQt5 import QtCore, QtGui, QtWidgets


def show_editor() -> None:
    """ Show the Funscript Editor Main Window """
    setup_logging()
    app = QtWidgets.QApplication(sys.argv)
    widget = FunscriptEditorWindow()
    widget.show()
    sys.exit(app.exec_())


def generate_funscript(video_file: str, start_time: int, output_file: str) -> None:
    """ Generate a funscript with minimal UI

    Args:
        video_file (str): path to video file
        start_time (int): start time in milliseconds
        output_file (str): path for the output file
    """
    setup_logging()
    app = QtWidgets.QApplication(sys.argv)
    generator = MinimalFunscriptGenerator(video_file, start_time, output_file)
    generator.run()
    sys.exit(app.exec_())
