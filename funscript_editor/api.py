import sys
import os
import cv2

import funscript_editor.utils.logging as logging
from funscript_editor.ui.funscript_editor_window import FunscriptEditorWindow
from funscript_editor.ui.funscript_generator_window import FunscriptGeneratorWindow
from funscript_editor.utils.config import VERSION

from PyQt5 import QtCore, QtGui, QtWidgets


def show_editor() -> None:
    """ Show the Funscript Editor Main Window """
    logging.setup_logging(silent=False)
    logger = logging.getLogger(__name__)
    logger.info("Python Funscript Editor %s", VERSION)
    logger.info("Startup Path: %s", str(os.getcwd()))
    app = QtWidgets.QApplication(sys.argv)
    widget = FunscriptEditorWindow()
    widget.show()
    ret = app.exec_()
    logger.info("EXIT")
    sys.exit(ret)


def generate_funscript(
        video_file: str,
        start_time: float,
        end_time :float,
        output_file: str,
        include_multiaxis_options: bool = False,
        enable_logging: bool = False) -> None:
    """ Generate a funscript with minimal UI

    Args:
        video_file (str): path to video file
        start_time (float): start time in milliseconds
        end_time (float): end time in milliseconds (set -1.0 to use video end)
        output_file (str): path for the output file
        include_multiaxis_options (bool): include options for multiaxis output
    """
    if enable_logging:
        logging.setup_logging(silent=True)
    else:
        logging.disable_logging()
    logger = logging.getLogger(__name__)
    logger.info("Python Funscript Generator %s", VERSION)
    logger.info("Startup Path: %s", str(os.getcwd()))
    logger.info("Args: video_file=%s, start_time=%s, end_time=%s, output_file=%s", \
            str(video_file), str(start_time), str(end_time), str(output_file))
    app = QtWidgets.QApplication(sys.argv)
    _ = FunscriptGeneratorWindow(video_file, start_time, end_time, output_file, include_multiaxis_options)
    ret = app.exec_()
    logger.info("EXIT")
    sys.exit(ret)
