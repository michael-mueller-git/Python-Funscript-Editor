import sys
import funscript_editor.utils.logging as logging
import os
import json
import time
import platform
import cv2

from funscript_editor.utils.logging import setup_logging
from funscript_editor.algorithms.trackingmanager import TrackingManagerThread, TrackingManagerParameter
from funscript_editor.data.funscript import Funscript
from funscript_editor.ui.settings_dialog import SettingsDialog
import funscript_editor.definitions as definitions
from funscript_editor.ui.theme import setup_theme
from funscript_editor.utils.config import SETTINGS, PROJECTION, HYPERPARAMETER
from funscript_editor.algorithms.scale import ScalingUiThread, ScalingUiParameter
from funscript_editor.ui.cut_tracking_result import CutTrackingResultWidget
from funscript_editor.data.ffmpegstream import FFmpegStream
from funscript_editor.ui.postprocessing import PostprocessingWidget

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
        include_multiaxis (bool): include multiaxis output
        no_tracking (bool): Use previous tracking result
    """

    def __init__(self,
            video_file: str,
            start_time: float,
            end_time: float,
            output_file: str,
            include_multiaxis: bool = False,
            no_tracking: bool = False):
        super(FunscriptGeneratorWindow, self).__init__()
        self.allow_close = False
        self.raw_output = False
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
                    + "(Make sure there are no special symbols in the video file path " \
                    + "-> Rename your video file and store them in a path without special characters you should also try to restart OFS)",
                    error=True)
            sys.exit()

        if not os.path.exists(video_file):
            self.__show_message("Video file not found ({})".format(video_file), error=True)
            sys.exit()

        self.video_info = FFmpegStream.get_video_info(video_file)
        self.video_file = video_file
        self.output_file = output_file
        self.start_frame = FFmpegStream.millisec_to_frame(start_time, self.video_info.fps)
        self.end_frame = FFmpegStream.millisec_to_frame(end_time, self.video_info.fps) if end_time > 0.0 and start_time < end_time else -1

        self.__logger.info("Set End Time to Frame Number %d", self.end_frame)
        self.__logger.info("Hyperparameter:" + str(HYPERPARAMETER))
        self.__logger.info("Config:" + str(SETTINGS))

        if no_tracking:
            self.continue_with_tracking_result()
        else:
            self.settings = {}
            self.settings_dialog = SettingsDialog(self.settings, include_vr = True, include_multiaxis = include_multiaxis)
            self.settings_dialog.applySettings.connect(self.run)
            self.settings_dialog.show()


    __logger = logging.getLogger(__name__)

    funscriptCompleted = QtCore.pyqtSignal(object, str, bool)
    openCutWidget = QtCore.pyqtBoundSignal


    def continue_with_tracking_result(self):
        self.__logger.info("Use previous tracking result")
        try:
            if not os.path.exists(definitions.RAW_TRACKING_DATA_CAHCE_FILE):
                self.__show_message("Tracking result not found")
                sys.exit()

            with open(definitions.RAW_TRACKING_DATA_CAHCE_FILE, 'r') as fd:
                cache_content = json.load(fd)

            if any(x not in cache_content for x in ["videoFile", "fps", "actions"]):
                self.__show_message("Invalid tracking result cache file")
                sys.exit()

            current_video_filename = os.path.basename(self.video_file)
            if cache_content["videoFile"] != current_video_filename:
                self.__show_message(f"Tracking result for {current_video_filename} not found")
                sys.exit()

            if cache_content["fps"] != self.video_info.fps:
                self.__show_message(f"Video propberies has changed")
                sys.exit()

            self.funscripts = {metric: Funscript(self.video_info.fps) \
                for metric in cache_content["actions"]}

            self.score = {metric: [ \
                    item["pos"] \
                    for item in cache_content["actions"][metric] \
                ] \
                for metric in cache_content["actions"] }

            start_time = min([cache_content["actions"][metric][0]["at"] for metric in cache_content["actions"]])
            self.start_frame = FFmpegStream.millisec_to_frame(float(start_time), self.video_info.fps)
            self.__logger.info("Set start frame to %d", self.start_frame)
        except Exception as ex:
            self.logger.critical("The program crashed in continue_with_tracking_result due to a fatal error", exc_info=ex)
            self.__show_message("Invalid previous tracking data")
            sys.exit()

        self.__next_postprocessing(None, [], [])


    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
        else:
            event.ignore()


    def get_absolute_framenumber(self, frame_number: int) -> int:
        """ Get the absoulte frame number

        Args:
            frame_number (int): relative frame number

        Returns:
            int: absolute frame position
        """
        return self.start_frame + frame_number


    def __show_message(self, message :str, error: bool = False) -> None:
        if error: self.__logger.error(message)
        else: self.__logger.info(message)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message+' ')
        msg.setWindowTitle("Funscript Generator " + str("Error" if error else "Info"))
        msg.exec_()


    # Step 1
    def __tracking_completed(self, score, projection_config, tracking_points, msg, success) -> None:
        self.funscripts = {k.replace('inverted', '').strip(): Funscript(self.video_info.fps) \
                for k in self.settings['trackingMetrics'].split('+')}

        if not success:
            self.__funscript_generated(self.funscripts, msg, success)
            return

        self.score = score
        self.projection_config = projection_config
        self.tracking_points = tracking_points
        self.msg = msg

        self.cut_tracking_result_widget = CutTrackingResultWidget(self.score, [k for k in self.funscripts])
        self.cut_tracking_result_widget.cutCompleted.connect(self.__cut_completed)
        if "open_postprocessing_maximized" in SETTINGS and SETTINGS["open_postprocessing_maximized"]: 
            self.__logger.info("open cut window as maximized window")
            self.cut_tracking_result_widget.showMaximized()
        else:
            self.cut_tracking_result_widget.show()


    # Step 2
    def __cut_completed(self, cut_values):
        self.__logger.info('cut values: %s', str(cut_values))
        self.start_frame += cut_values["start"]
        for k in self.score:
            try:
                self.score[k] = self.score[k][cut_values["start"]:cut_values["stop"]]
            except:
                pass
        self.scaling = ScalingUiThread(
                self.video_info,
                ScalingUiParameter(
                    video_path = self.video_file,
                    projection_config = self.projection_config,
                    metrics = [k for k in self.funscripts],
                    start_frame = self.start_frame,
                    end_frame = self.end_frame,
            ), self.score, self.tracking_points, self.msg)

        self.scaling.scalingCompleted.connect(self.__scaling_completed)
        self.scaling.start()


    # Step 3
    def __scaling_completed(self, score):
        self.score = score
        self.__logger.info('scaling completed')

        # 1. Now first save the raw tracking data

        for metric in self.funscripts:
            for idx, val in enumerate(self.score[metric]):
                self.funscripts[metric].add_action(
                        round(val),
                        FFmpegStream.frame_to_millisec(self.get_absolute_framenumber(idx), self.video_info.fps),
                        True
                    )

        raw_tracking_data_json_output = {
            'version': 1,
            'comment': "MTFG RAW DATA",
            'fps': self.video_info.fps,
            'videoFile': os.path.basename(self.video_file),
            'actions': {}
        }

        for key in self.funscripts:
            raw_tracking_data_json_output['actions'][key] = []
            for item in self.funscripts[key].get_actions():
                raw_tracking_data_json_output['actions'][key].append(item)

        os.makedirs(definitions.CACHE_DIR, exist_ok=True)
        with open(definitions.RAW_TRACKING_DATA_CAHCE_FILE, 'w') as f:
            json.dump(raw_tracking_data_json_output, f)

        # 2. Continue with selected user option on raw data

        if not self.raw_output:
            self.__logger.info("Post Processing Data")
            #NOTE: delete the raw data from funscripts
            for key in self.funscripts:
                self.funscripts[key].clear_actions()
            self.__next_postprocessing(None, [], [])
        else:
            self.__logger.info("Raw Output")
            self.__funscript_generated(self.funscripts, "OK", True)


    # Setp 4
    def __next_postprocessing(self, last_metric, idx_keep, val_keep):
        if last_metric is not None:
            self.__logger.info("apply score %s with %d idx", last_metric, len(idx_keep))
            for idx, val in zip(idx_keep, val_keep):
                self.funscripts[last_metric].add_action(
                        round(val),
                        FFmpegStream.frame_to_millisec(self.get_absolute_framenumber(idx), self.video_info.fps),
                        True
                    )

        found_last = last_metric is None
        for metric in self.funscripts:
            if metric == last_metric:
                found_last = True
                continue

            if found_last:
                self.postprocessing_widget = PostprocessingWidget(metric, self.score[metric], self.video_info)
                self.postprocessing_widget.postprocessingCompleted.connect(self.__next_postprocessing)
                if "open_postprocessing_maximized" in SETTINGS and SETTINGS["open_postprocessing_maximized"]: 
                    self.__logger.info("open postprocessing as maximized window")
                    self.postprocessing_widget.showMaximized()
                else:
                    self.postprocessing_widget.show()
                return

        self.__funscript_generated(self.funscripts, "OK", True)


    # Setp 5
    def __funscript_generated(self, funscripts, msg, success) -> None:
        self.allow_close = True
        first_metric = [x for x in funscripts.keys()][0]

        if isinstance(self.output_file, Funscript):
            if len(funscripts) > 1:
                self.__logger.warning("Multiaxis output for build-in UI is not implemented")
            for item in funscripts[first_metric].get_actions():
                self.output_file.add_action(item['pos'], item['at'], True)
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
                self.__logger.info("EXIT with Workaround")
                time.sleep(0.5)
                os.system("taskkill /f /im funscript-editor.exe")

            # os._exit(os.EX_OK)
            sys.exit()


    def run(self) -> None:
        """ start generator """
        self.__logger.info('settings: %s', str(self.settings))
        self.settings['videoType'] = list(filter(lambda x: PROJECTION[x]['name'] == self.settings['videoType'], PROJECTION.keys()))[0]
        self.metrics = {k.replace('inverted', '').strip(): {"inverted": "inverted" in k} for k in self.settings['trackingMetrics'].split('+')}
        self.raw_output = self.settings["outputMode"] == "normalized raw tracking data"
        self.tracking_manager = TrackingManagerThread(
                self.video_info,
                TrackingManagerParameter(
                    video_path = self.video_file,
                    projection = self.settings['videoType'],
                    start_frame = self.start_frame,
                    track_men = 'two' in self.settings['trackingMethod'],
                    supervised_tracking = 'Supervised' in self.settings['trackingMethod'],
                    tracking_metrics = self.metrics,
                    end_frame = self.end_frame,
                    number_of_trackers = int(self.settings['numberOfTracker']),
                    supervised_tracking_is_exit_condition = "stopping" in self.settings['trackingMethod'],
                    skip_frames = int(self.settings['processingSpeed']),
                ))

        self.tracking_manager.trackingCompleted.connect(self.__tracking_completed)
        self.tracking_manager.start()
