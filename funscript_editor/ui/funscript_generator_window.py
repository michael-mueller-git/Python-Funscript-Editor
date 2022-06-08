import sys
import funscript_editor.utils.logging as logging
import os
import json
import time
import platform
import cv2

from funscript_editor.utils.logging import setup_logging
from funscript_editor.algorithms.funscriptgenerator import FunscriptGeneratorThread, FunscriptGeneratorParameter
from funscript_editor.data.funscript import Funscript
from funscript_editor.ui.settings_dialog import SettingsDialog
import funscript_editor.definitions as definitions
from funscript_editor.ui.theme import setup_theme
from funscript_editor.utils.config import SETTINGS, PROJECTION, HYPERPARAMETER

from PyQt5 import QtCore, QtGui, QtWidgets

USE_OPTICALFLOW = False # Enable some hardcoded optical flow testcode
if USE_OPTICALFLOW:
    from funscript_editor.algorithms.opticalflow import OpticalFlowFunscriptGeneratorThread, OpticalFlowFunscriptGeneratorParameter

class FunscriptGeneratorWindow(QtWidgets.QMainWindow):
    """ Class to Generate a funscript with minimal UI

    Note:
        We us a hidden QMainWindow to get all Qt functions like QThread and QSignal.

    Args:
        video_file (str): path to video file
        start_time (float): start position in video (timestamp in milliseconds)
        end_time (float): end position in video (timestamp in milliseconds) use -1.0 for video end.
        output_file (str, Funscript): csv output file path (Optional you can pass a funscript object where to store the result)
        include_multiaxis (bool): include multiaxis output
    """

    def __init__(self,
            video_file: str,
            start_time: float,
            end_time: float,
            output_file: str,
            include_multiaxis: bool = False):
        super(FunscriptGeneratorWindow, self).__init__()
        setup_theme()
        if os.path.exists(definitions.ICON_PATH):
            self.setWindowIcon(QtGui.QIcon(definitions.ICON_PATH))

        if include_multiaxis:
            self.__logger.info("Enable multiaxis output")

        if not isinstance(output_file, Funscript):
            output_file = os.path.abspath(output_file)
            if os.path.isdir(output_file):
                self.__show_message("The output TempFile path must be a file not a folder", error=True)
                sys.exit()

            if os.path.exists(output_file):
                os.remove(output_file)

        if video_file is None or video_file == "":
            self.__show_message("Video file was not specified! " \
                    + "(Make sure there are no special symbols like squared brackets in the video file path " \
                    + "-> Rename your video file and store them in a path without special characters)",
                    error=True)
            sys.exit()

        if not os.path.exists(video_file):
            self.__show_message("Video file not found ({})".format(video_file), error=True)
            sys.exit()

        cap = cv2.VideoCapture(video_file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_aspect_ratio = float(width) / max((1, float(height)))
        cap.release()

        self.video_file = video_file
        self.is_sbs_vr_video = True if 1.9 < video_aspect_ratio < 2.1 else False
        self.output_file = output_file

        self.start_frame = int(round(float(start_time)/(float(1000)/float(self.fps)))) if start_time > 0.0 else 0
        self.end_frame = int(round(float(end_time)/(float(1000)/float(self.fps)))) if end_time > 0.0 and start_time < end_time else -1

        self.__logger.info("Set End Time to Frame Number %d", self.end_frame)

        self.__logger.info("Hyperparameter:" + str(HYPERPARAMETER))
        self.__logger.info("Config:" + str(SETTINGS))

        self.settings = {}
        if USE_OPTICALFLOW:
            self.run_opticalflow()
        else:
            self.settings_dialog = SettingsDialog(self.settings, include_vr = True, include_multiaxis = include_multiaxis)
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


    def __funscript_generated(self, funscripts, msg, success) -> None:
        first_metric = [x for x in funscripts.keys()][0]

        if isinstance(self.output_file, Funscript):
            if len(funscripts) > 1:
                self.__logger.warning("Multiaxis output for build-in UI is not implemented")
            for item in funscripts[first_metric].get_actions():
                self.output_file.add_action(item['pos'], item['at'], SETTINGS['raw_output'])
            self.funscriptCompleted.emit(self.output_file, msg, success)
        else:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            if self.output_file.lower().endswith('.json'):
                funscript_json_output = {
                        'version': 1,
                        'actions': {}
                    }

                for key in funscripts.keys():
                    funscript_json_output['actions'][key] = []
                    for item in funscripts[key].get_actions():
                        funscript_json_output['actions'][key].append(item)

                with open(self.output_file, 'w') as f:
                    json.dump(funscript_json_output, f)
                    f.flush()
            else:
                # dump to CSV
                if len(funscripts) > 1:
                    self.__logger.warning("Multiaxis output for csv is  not implemented")
                with open(self.output_file, 'w') as f:
                    f.write('at;pos\n')
                    for item in funscripts[first_metric].get_actions():
                        f.write('{at};{pos}\n'.format(at=item['at'], pos=item['pos']))

                    f.flush()

            self.__logger.info("Save result to %s", self.output_file)
            if not success: self.__show_message(msg, error=True)
            if platform.system() == 'Windows':
                time.sleep(0.5)
                os.system("taskkill /f /im funscript-editor.exe")

            # os._exit(os.EX_OK)
            sys.exit()


    def run_opticalflow(self) -> None:
        self.funscripts = {'movement': Funscript(self.fps)}
        self.funscript_generator = OpticalFlowFunscriptGeneratorThread(
                OpticalFlowFunscriptGeneratorParameter(
                    video_path = self.video_file,
                    projection = "vr_he_180_sbs",
                    start_frame = self.start_frame,
                    end_frame = self.end_frame,
                    skip_frames = 0
                    ),
                self.funscripts)
        self.funscript_generator.funscriptCompleted.connect(self.__funscript_generated)
        self.funscript_generator.start()


    def run(self) -> None:
        """ start generator """
        self.__logger.info('settings: %s', str(self.settings))
        self.settings['videoType'] = list(filter(lambda x: PROJECTION[x]['name'] == self.settings['videoType'], PROJECTION.keys()))[0]
        self.funscripts = {k.replace('inverted', '').strip(): Funscript(self.fps, inverted = "inverted" in k) for k in self.settings['trackingMetrics'].split('+')}
        self.funscript_generator = FunscriptGeneratorThread(
                FunscriptGeneratorParameter(
                    video_path = self.video_file,
                    track_men = 'two' in self.settings['trackingMethod'],
                    supervised_tracking = 'Supervised' in self.settings['trackingMethod'],
                    supervised_tracking_is_exit_condition = "stopping" in self.settings['trackingMethod'],
                    projection = self.settings['videoType'],
                    start_frame = self.start_frame,
                    end_frame = self.end_frame,
                    number_of_trackers = int(self.settings['numberOfTracker']),
                    points = self.settings['points'].lower().replace(' ', '_'),
                    additional_points = self.settings['additionalPoints'].lower().replace(' ', '_'),
                    skip_frames = int(self.settings['processingSpeed']),
                    top_points_offset = self.settings['topPointOffset'],
                    bottom_points_offset = self.settings['bottomPointOffset']
                ),
                self.funscripts)

        self.funscript_generator.funscriptCompleted.connect(self.__funscript_generated)
        self.funscript_generator.start()
