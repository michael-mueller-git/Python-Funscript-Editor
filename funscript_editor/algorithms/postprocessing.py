import copy
from dataclasses import dataclass
import funscript_editor.utils.logging as logging
from funscript_editor.data.ffmpegstream import FFmpegStream
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS
from funscript_editor.data.funscript import Funscript
from funscript_editor.algorithms.signal import Signal
from funscript_editor.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters
import numpy as np

@dataclass
class PostprocessingParameter:
    """ Postprocessing Parameter Dataclass with default values """
    video_path: str
    projection_config: dict
    start_frame: int
    end_frame: int = -1 # default is video end (-1)
    skip_frames: int = 1
    raw_output: bool = SETTINGS["raw_output"]

    # remoed later:
    points: str = "local_min_max"
    additional_points: str = "none"
    top_points_offset: float = 15.0
    bottom_points_offset: float = -15.0

class Postprocessing:

    def __init__(self,
                 params: PostprocessingParameter,
                 score: dict,
                 tracking_points: dict,
                 funscripts: dict,
                 msg: str):
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.video_info = FFmpegStream.get_video_info(self.params.video_path)
        self.score = score
        self.tracking_points = tracking_points
        self.funscripts = funscripts
        self.msg = msg
        self.ui = OpenCV_GUI(OpenCV_GUI_Parameters(
            video_info = self.video_info,
            skip_frames = self.params.skip_frames,
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


    def create_funscript(self, idx_dict: dict, metric: str) -> None:
        """ Generate the Funscript

        Args:
            idx_dict (dict): dictionary with all local max and min points in score
                             {'min':[idx1, idx2, ...], 'max':[idx1, idx2, ...]}
            metric: metric key
        """
        if metric not in self.funscripts.keys():
            self.logger.error("metric %s not in funscripts keys", metric)
            return

        if self.params.raw_output:
            output_score = copy.deepcopy(self.score[metric])
            self.logger.warning("Insert %d raw points", len(output_score))
            for idx in range(len(output_score)):
                self.funscripts[metric].add_action(
                        round(output_score[idx]),
                        FFmpegStream.frame_to_millisec(self.get_absolute_framenumber(idx), self.video_info.fps),
                        True
                    )

        else:
            output_score = self.get_score_with_offset(idx_dict, metric)

            for idx in idx_dict['min']:
                self.funscripts[metric].add_action(
                        round(output_score[idx]),
                        FFmpegStream.frame_to_millisec(self.get_absolute_framenumber(idx), self.video_info.fps)
                    )

            for idx in idx_dict['max']:
                self.funscripts[metric].add_action(
                        round(output_score[idx]),
                        FFmpegStream.frame_to_millisec(self.get_absolute_framenumber(idx), self.video_info.fps)
                    )


    def determine_change_points(self, metric: str) -> dict:
        """ Determine all change points

        Args:
            metric (str): from which metric you want to have the chainge points

        Returns:
            dict: all local max and min points in score {'min':[idx1, idx2, ...], 'max':[idx1, idx2, ...]}
        """
        self.logger.info("Determine change points for %s", metric)
        if metric not in self.score.keys():
            self.logger.error("key %s not in score metrics dict", metric)
            return dict()

        base_point_algorithm = Signal.BasePointAlgorithm.local_min_max
        if self.params.points == 'direction_changed':
            base_point_algorithm = Signal.BasePointAlgorithm.direction_changes

        additional_points_algorithms = []
        if self.params.additional_points == 'high_second_derivative':
            additional_points_algorithms.append(Signal.AdditionalPointAlgorithm.high_second_derivative)

        if self.params.additional_points == 'distance_minimization':
            additional_points_algorithms.append(Signal.AdditionalPointAlgorithm.distance_minimization)

        if self.params.additional_points == 'evenly_intermediate':
            additional_points_algorithms.append(Signal.AdditionalPointAlgorithm.evenly_intermediate)

        signal = Signal(self.video_info.fps)
        decimate_indexes = signal.decimate(
                self.score[metric],
                base_point_algorithm,
                additional_points_algorithms
        )
        return signal.categorize_points(self.score[metric], decimate_indexes)


    def get_score_with_offset(self, idx_dict: dict, metric: str) -> list:
        """ Apply the offsets form settings dialog

        Args:
            idx_dict (dict): the idx dictionary with {'min':[], 'max':[]} idx lists
            metric (str): the metric for the score calculation

        Returns:
            list: score with offset
        """
        offset_max = self.params.top_points_offset
        offset_min = self.params.bottom_points_offset

        score = copy.deepcopy(self.score[metric])
        score_min, score_max = min(score), max(score)

        for idx in idx_dict['max']:
            score[idx] = max(( score_min, min((score_max, score[idx] + offset_max)) ))

        for idx in idx_dict['min']:
            score[idx] = max(( score_min, min((score_max, score[idx] + offset_min)) ))

        return score


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


    def run(self) -> dict:
        for metric in self.funscripts:
            # self.scale_score(self.msg, metric=metric)
            idx_dict = self.determine_change_points(metric)
            self.create_funscript(idx_dict, metric)

        self.ui.close()
        return self.funscripts
