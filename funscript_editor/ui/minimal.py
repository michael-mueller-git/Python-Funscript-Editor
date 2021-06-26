import sys
import logging
import os
import cv2

from funscript_editor.utils.logging import setup_logging
from funscript_editor.ui.funscript_editor_window import FunscriptEditorWindow
from funscript_editor.algorithms.funscriptgenerator import FunscriptGenerator, FunscriptGeneratorParameter
from funscript_editor.data.funscript import Funscript

from PyQt5 import QtCore, QtGui, QtWidgets


class MinimalFunscriptGenerator(QtWidgets.QMainWindow):
    """ Class to Generate a funscript with minimal UI

    Note:
        We us a hidden QMainWindow to get all Qt functions like QThread and QSignal.

    Args:
        video_file (str): path to video file
        start_time (int): start position in video (timestamp in milliseconds)
    """

    def __init__(self, video_file, start_time, output_file):
        super(MinimalFunscriptGenerator, self).__init__()

        if os.path.isdir(output_file):
            self.__logger.error("The output TempFile path must be a file not a folder")
            self.__show_message("The output TempFile path must be a file not a folder")
            sys.exit()

        if os.path.exists(output_file):
            os.remove(output_file)

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.funscript = Funscript(fps)
        self.output_file = os.path.abspath(output_file)

        if False:
            reply = QtWidgets.QMessageBox.question(None, 'Generate Funscript ', \
                    'Do you want to generate the funscript actions by tracking features in the Video? ',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                logging.info('abort')
                sys.exit()

        if video_file is None or video_file == "":
            self.__logger.error("Video file was not specified!")
            self.__show_message("Video file was not specified!")
            sys.exit()

        if not os.path.exists(video_file):
            self.__logger.error("Video file not found: %s", video_file)
            self.__show_message("Video file not found ({})".format(video_file))
            sys.exit()

        reply = QtWidgets.QMessageBox.question(None, 'Track Men', 'Do you want to track the Men? ',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        trackMen = True if reply == QtWidgets.QMessageBox.Yes else False

        start_frame = int(round(float(start_time)/(float(1000)/float(fps))))

        self.funscript_generator = FunscriptGenerator(
                FunscriptGeneratorParameter(
                    video_path = video_file,
                    start_frame = start_frame,
                    track_men = trackMen
                ),
                self.funscript)
        self.funscript_generator.funscriptCompleted.connect(self.__funscript_generated)
        self.funscript_generator.processStatus.connect(self.__status_changed)


    __logger = logging.getLogger(__name__)


    def __show_message(self, message :str) -> None:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message+' ')
        msg.setWindowTitle("Funscript Generator Info")
        msg.exec_()


    def __funscript_generated(self, funscript, msg, success) -> None:
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            f.write('at;pos\n')
            for item in funscript.get_actions():
                f.write('{at};{pos}\n'.format(at=item['at'], pos=item['pos']))
        logging.info('Completed with status message: "%s"', msg)
        if not success: self.__show_message(msg)
        sys.exit()


    def __status_changed(self, frame) -> None:
        pass


    def run(self) -> None:
        """ start generator """
        self.funscript_generator.start()
