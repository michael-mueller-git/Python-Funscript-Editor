import sys
import logging
import os
import cv2

from funscript_editor.utils.logging import setup_logging
from funscript_editor.ui.funscript_editor_window import FunscriptEditorWindow
from funscript_editor.ui.funscript_generator_window import FunscriptGeneratorWindow
from funscript_editor.utils.config import VERSION

from PyQt5 import QtCore, QtGui, QtWidgets


def show_editor() -> None:
    """ Show the Funscript Editor Main Window """
    setup_logging()
    logging.info("Python Funscript Editor %s", VERSION)
    logging.info("Startup Path: %s", str(os.getcwd()))
    app = QtWidgets.QApplication(sys.argv)
    widget = FunscriptEditorWindow()
    widget.show()
    sys.exit(app.exec_())


def generate_funscript(
        video_file: str,
        start_time: float,
        end_time :float,
        output_file: str) -> None:
    """ Generate a funscript with minimal UI

    Args:
        video_file (str): path to video file
        start_time (float): start time in milliseconds
        end_time (float): end time in milliseconds (set -1.0 to use video end)
        output_file (str): path for the output file
    """
    setup_logging()
    logging.info("Python Funscript Generator %s", VERSION)
    logging.info("Startup Path: %s", str(os.getcwd()))
    logging.info("Args: video_file=%s, start_time=%s, end_time=%s, output_file=%s", \
            str(video_file), str(start_time), str(end_time), str(output_file))
    app = QtWidgets.QApplication(sys.argv)
    generator = FunscriptGeneratorWindow(video_file, start_time, end_time, output_file)
    sys.exit(app.exec_())
