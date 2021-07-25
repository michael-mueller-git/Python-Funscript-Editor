import sys
import logging
import os
import cv2

from funscript_editor.utils.logging import setup_logging
from funscript_editor.algorithms.funscriptgenerator import FunscriptGeneratorThread, FunscriptGeneratorParameter
from funscript_editor.data.funscript import Funscript
from funscript_editor.ui.settings_dialog import SettingsDialog

from PyQt5 import QtCore, QtGui, QtWidgets


class FunscriptGeneratorWindow(QtWidgets.QMainWindow):
    """ Class to Generate a funscript with minimal UI

    Note:
        We us a hidden QMainWindow to get all Qt functions like QThread and QSignal.

    Args:
        video_file (str): path to video file
        start_time (float): start position in video (timestamp in milliseconds)
        end_time (float): end position in video (timestamp in milliseconds) use -1.0 for video end.
        output_file (str, Funscript): csv output file path (Optional you can pass a funscript object where to store the result)
    """

    def __init__(self,
            video_file: str,
            start_time: float,
            end_time: float,
            output_file: str):
        super(FunscriptGeneratorWindow, self).__init__()

        if not isinstance(output_file, Funscript):
            output_file = os.path.abspath(output_file)
            if os.path.isdir(output_file):
                self.__show_message("The output TempFile path must be a file not a folder", error=True)
                sys.exit()

            if os.path.exists(output_file):
                os.remove(output_file)

        if video_file is None or video_file == "":
            self.__show_message("Video file was not specified! " \
                    + "(Make sure there are no special symbols like squared brackets in the video file path)",
                    error=True)
            sys.exit()

        if not os.path.exists(video_file):
            self.__show_message("Video file not found ({})".format(video_file), error=True)
            sys.exit()

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.video_file = video_file
        self.funscript = Funscript(fps)
        self.output_file = output_file

        self.start_frame = int(round(float(start_time)/(float(1000)/float(fps)))) if start_time > 0.0 else 0
        self.end_frame = int(round(float(end_time)/(float(1000)/float(fps)))) if end_time > 0.0 and start_time < end_time else -1

        self.__logger.info("Set End Time to Frame Number %d", self.end_frame)

        self.settings = {}
        self.settings_dialog = SettingsDialog(self.settings)
        self.settings_dialog.applySettings.connect(self.run)
        self.settings_dialog.show()


    __logger = logging.getLogger(__name__)

    funscriptCompleted = QtCore.pyqtSignal(object, str, bool)


    def __show_message(self, message :str, error: bool = False) -> None:
        if error: self.__logger.error(message)
        else: self.__logger.info(message)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message+' ')
        msg.setWindowTitle("Funscript Generator " + str("Error" if error else "Info"))
        msg.exec_()


    def __funscript_generated(self, funscript, msg, success) -> None:
        if isinstance(self.output_file, Funscript):
            for item in funscript.get_actions():
                self.output_file.add_action(item['pos'], item['at'])
            self.funscriptCompleted.emit(self.output_file, msg, success)
        else:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w') as f:
                f.write('at;pos\n')
                for item in funscript.get_actions():
                    f.write('{at};{pos}\n'.format(at=item['at'], pos=item['pos']))

            self.__logger.info("Save result to %s", self.output_file)
            if not success: self.__show_message(msg, error=True)
            sys.exit()


    def run(self) -> None:
        """ start generator """
        self.__logger.info('settings: %s', str(self.settings))
        self.funscript_generator = FunscriptGeneratorThread(
                FunscriptGeneratorParameter(
                    video_path = self.video_file,
                    track_men = self.settings['trackingMethod'] != 'Woman',
                    direction = self.settings['trackingAxis'],
                    projection = self.settings['videoType'],
                    start_frame = self.start_frame,
                    end_frame = self.end_frame
                ),
                self.funscript)
        self.funscript_generator.funscriptCompleted.connect(self.__funscript_generated)
        self.funscript_generator.start()
