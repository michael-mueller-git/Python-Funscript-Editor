""" The Opencv ui need to run in a seperate QtTread """
import copy
from dataclasses import dataclass
import funscript_editor.utils.logging as logging
from funscript_editor.data.ffmpegstream import FFmpegStream
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS
from funscript_editor.data.funscript import Funscript
from funscript_editor.algorithms.signal import Signal
from funscript_editor.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters
from funscript_editor.ui.cut_tracking_result import CutTrackingResultWidget
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets

@dataclass
class ScalingUiParameter:
    """ Postprocessing Parameter Dataclass with default values """
    video_path: str
    projection_config: dict
    metrics: list
    start_frame: int
    end_frame: int = -1 # default is video end (-1)


class ScalingUiThread(QtCore.QThread):

    #: completed event with funscripts, status message and success flag
    scalingCompleted = QtCore.pyqtSignal(dict)

    def __init__(self,
                 video_info,
                 params: ScalingUiParameter,
                 score: dict,
                 tracking_points: dict,
                 msg: str):
        QtCore.QThread.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.video_info = video_info
        self.params = params
        self.score = score
        self.tracking_points = tracking_points
        self.msg = msg
        self.ui = OpenCV_GUI(OpenCV_GUI_Parameters(
            video_info = self.video_info,
            skip_frames = 0,
            end_frame_number = self.params.end_frame
            ))


    def get_absolute_framenumber(self, frame_number: int) -> int:
        """ Get the absoulte frame number

        Args:
            frame_number (int): relative frame number

        Returns:
            int: absolute frame position
        """
        return self.params.start_frame + frame_number


    def get_tracking_points_by_frame_number(self, relative_frame_number: int) -> list:
        """ Get tracking points by frame number

        Args:
            relative_frame_number (int): relative frame number

        Returns:
            list: all tracking points
        """
        result = []
        for tracker_type in self.tracking_points.keys():
            for idx, tracker_number in enumerate([k for k in self.tracking_points[tracker_type].keys()]):
                if len(result) < idx + 1:
                    if relative_frame_number < len(self.tracking_points[tracker_type][tracker_number]):
                        result.append([self.tracking_points[tracker_type][tracker_number][relative_frame_number]])
                    else:
                        result.append([])
                else:
                    if relative_frame_number < len(self.tracking_points[tracker_type][tracker_number]):
                        result[idx].append(self.tracking_points[tracker_type][tracker_number][relative_frame_number])

        return result


    def scale_score(self, status: str, metric : str = 'y') -> None:
        """ Scale the score to desired stroke high

        Note:
            We determine the lowerst and highes positions in the score and request the real position from user.

        Args:
            status (str): a status/info message to display in the window
            metric (str): scale the 'y' or 'x' score
        """
        if metric not in self.score.keys():
            self.logger.error("key %s is not in score dict", metric)
            return

        if len(self.score[metric]) < 2: return
        min_frame = np.argmin(np.array(self.score[metric])) + self.params.start_frame
        max_frame = np.argmax(np.array(self.score[metric])) + self.params.start_frame

        success_min, success_max = True, True
        imgMin = FFmpegStream.get_frame(self.params.video_path, min_frame)
        imgMax = FFmpegStream.get_frame(self.params.video_path, max_frame)

        if success_min and success_max:
            imgMin = FFmpegStream.get_projection(imgMin, self.params.projection_config)
            imgMax = FFmpegStream.get_projection(imgMax, self.params.projection_config)

            min_tracking_points = self.get_tracking_points_by_frame_number(min_frame - self.params.start_frame)
            max_tracking_points = self.get_tracking_points_by_frame_number(max_frame - self.params.start_frame)

            for points in min_tracking_points:
                imgMin = OpenCV_GUI.draw_point_to_image(imgMin, points, connect_points=True)

            for points in max_tracking_points:
                imgMax = OpenCV_GUI.draw_point_to_image(imgMax, points, connect_points=True)

            # print('min_tracking_points', min_tracking_points, 'max_tracking_points', max_tracking_points)

            (desired_min, desired_max) = self.ui.min_max_selector(
                    image_min = imgMin,
                    image_max = imgMax,
                    info = status,
                    title_min = metric + " Minimum",
                    title_max = metric + " Maximum",
                    recommend_lower = round(min(self.score[metric])) if metric == 'roll' else 0,
                    recommend_upper = round(max(self.score[metric])) if metric == 'roll' else 99
                )
        else:
            self.logger.warning("Determine min and max failed")
            desired_min = 0
            desired_max = 99

        self.logger.info("Scale score %s to user input", metric)

        if metric == 'roll':
            self.score[metric] = Signal.scale_with_center(self.score[metric], desired_min, desired_max, 50)
        else:
            self.score[metric] = Signal.scale(self.score[metric], desired_min, desired_max)


    def run(self) -> None:
        for metric in self.params.metrics:
            self.scale_score(self.msg, metric=metric)

        self.ui.close()
        self.scalingCompleted.emit(self.score)
