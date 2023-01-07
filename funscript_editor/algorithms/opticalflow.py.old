from os import path
import cv2
import time
import json
import copy
import logging

import numpy as np

from funscript_editor.data.ffmpegstream import FFmpegStream
from dataclasses import dataclass
from PyQt5 import QtCore
from funscript_editor.algorithms.signal import Signal
from funscript_editor.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters
from funscript_editor.algorithms.ppca import PPCA

import matplotlib.pyplot as plt

@dataclass
class OpticalFlowFunscriptGeneratorParameter:
    """ Funscript Generator Parameter Dataclass with default values """
    video_path: str
    projection: str
    start_frame: int
    end_frame: int = -1 # default is video end (-1)
    skip_frames: int = 0
    min_trajectory_len: int = 40
    feature_detect_interval: int = 10
    movement_filter: float = 10.0


class OpticalFlowFunscriptGeneratorThread(QtCore.QThread):
    """ Funscript Generator Thread

    Args:
        params (OpticalFlowFunscriptGeneratorParameter): required parameter for the funscript generator
        funscript (dict): the references to the Funscript where we store the predicted actions
    """

    def __init__(self,
                 params: OpticalFlowFunscriptGeneratorParameter,
                 funscripts: dict):
        QtCore.QThread.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.funscripts = funscripts
        self.video_info = FFmpegStream.get_video_info(self.params.video_path)

        self.ui = OpenCV_GUI(OpenCV_GUI_Parameters(
            video_info = self.video_info,
            skip_frames = self.params.skip_frames,
            end_frame_number = self.params.end_frame
            ))


    #: completed event with reference to the funscript with the predicted actions, status message and success flag
    funscriptCompleted = QtCore.pyqtSignal(dict, str, bool)


    class OpticalFlowPyrLK:

        def __init__(self, min_trajectory_len, feature_detect_interval, feature_area):
            self.min_trajectory_len = min_trajectory_len
            self.feature_detect_interval = feature_detect_interval
            self.feature_area = feature_area
            self.trajectories = []
            self.frame_idx = 0
            self.prev_frame_gray = None
            self.result = []

            self.lk_params = dict(
                    winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

            self.feature_params = dict(
                    maxCorners = 20,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7
                )


        def update(self, frame_roi):
            frame_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            if len(self.trajectories) > 0:
                p0 = np.float32([trajectory[-1] for trajectory in self.trajectories]).reshape(-1, 1, 2)
                p1, _, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame_gray, frame_gray, p0, None, **self.lk_params)
                p0r, _, _ = cv2.calcOpticalFlowPyrLK(frame_gray, self.prev_frame_gray, p1, None, **self.lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1

                new_trajectories = []
                for trajectory, (x, y), good_flag in zip(self.trajectories, p1.reshape(-1, 2), good):
                    if not good_flag:
                        if len (trajectory) > self.min_trajectory_len:
                            # print('add trajectorie from', self.frame_idx - len(trajectory), 'to', self.frame_idx)
                            self.result.append({'end': self.frame_idx, 'trajectory': trajectory})
                        continue
                    trajectory.append((x, y))
                    new_trajectories.append(trajectory)

                self.trajectories = new_trajectories


            if len(self.trajectories) == 0 or self.frame_idx % self.feature_detect_interval == 0:
                seach_img = frame_gray[self.feature_area[1]:self.feature_area[1]+self.feature_area[3], self.feature_area[0]:self.feature_area[0]+self.feature_area[2]]
                mask = np.zeros_like(seach_img)
                mask[:] = 255
                p = cv2.goodFeaturesToTrack(seach_img, mask = mask, **self.feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        x += self.feature_area[0]
                        y += self.feature_area[1]
                        if any(abs(t[-1][0] - x) < 3 and abs(t[-1][1] - y) < 3 for t in self.trajectories):
                            continue

                        self.trajectories.append([(x, y)])


            self.frame_idx += 1
            self.prev_frame_gray = frame_gray

            return [t[-1] for t in self.trajectories]


        def get_result(self):
            result = copy.deepcopy(self.result)
            for trajectory in self.trajectories:
                if len (trajectory) > (self.min_trajectory_len / 2):
                    result.append({'end': self.frame_idx, 'trajectory': trajectory})

            return { 'meta': { 'last_idx': self.frame_idx }, 'data': result }


    def extract_movement(self, optical_flow_result, metric_idx = 1, filter_static_points = True):
        result = []
        for r in optical_flow_result['data']:
            zero_before = r['end'] - len(r['trajectory'])
            zero_after = optical_flow_result['meta']['last_idx'] - r['end']
            trajectory_min = min([item[metric_idx] for item in r['trajectory']])
            action = [(r['trajectory'][i][metric_idx] - trajectory_min) for i in range(len(r['trajectory']))]
            y = [None for _ in range(zero_before)] + action + [None for _ in range(zero_after)]
            if not filter_static_points or (max(action) - min(action)) > self.params.movement_filter:
                result.append(y)

        return result


    def get_absolute_framenumber(self, frame_number: int) -> int:
        """ Get the absoulte frame number

        Note:
            We have an offset of 1 because we use the first frame for init

        Args:
            frame_number (int): relative frame number

        Returns:
            int: absolute frame position
        """
        return self.params.start_frame + frame_number + 1


    def tracking(self) -> str:
        """ Tracking function to track the features in the video

        Returns:
            str: a process status message e.g. 'end of video reached'
        """
        first_frame = FFmpegStream.get_frame(self.params.video_path, self.params.start_frame)

        projection_config = self.ui.get_video_projection_config(first_frame, self.params.projection)

        video = FFmpegStream(
                video_path = self.params.video_path,
                config = projection_config,
                skip_frames = self.params.skip_frames,
                start_frame = self.params.start_frame
            )

        first_frame = video.read()
        if first_frame is None:
            return "FFmpeg could not extract the first video frame"

        preview_frame = copy.copy(first_frame)
        search_roi = self.ui.bbox_selector(
            preview_frame,
            "Select observe area of an single person",
        )

        preview_frame = self.ui.draw_box_to_image(
                preview_frame,
                search_roi,
                color=(0,255,0)
            )

        while True:
            feature_roi = self.ui.bbox_selector(
                preview_frame,
                "Select feature area inside the observe area",
            )

            if feature_roi[0] > search_roi[0] \
                and feature_roi[1] > search_roi[1] \
                and feature_roi[0] + feature_roi[2] < search_roi[0] + search_roi[2] \
                and feature_roi[1] + feature_roi[3] < search_roi[1] + search_roi[3]:
                break

            self.logger.warning("Invalid feature")

        feature_roi = [
                feature_roi[0] - search_roi[0],
                feature_roi[1] - search_roi[1],
                feature_roi[2],
                feature_roi[3]
            ]

        optical_flow = OpticalFlowFunscriptGeneratorThread.OpticalFlowPyrLK(
                min_trajectory_len = self.params.min_trajectory_len,
                feature_detect_interval = self.params.feature_detect_interval,
                feature_area = feature_roi
            )

        status = "End of video reached"
        frame_num = 1 # first frame is init frame
        while video.isOpen():
            frame = video.read()
            frame_num += (self.params.skip_frames+1)

            if frame is None:
                status = 'Reach a corrupt video frame' if video.isOpen() else 'End of video reached'
                break

            if self.params.end_frame > 0 and frame_num + self.params.start_frame >= self.params.end_frame:
                status = "Tracking stop at existing action point"
                break

            frame_roi = frame[search_roi[1]:search_roi[1]+search_roi[3], search_roi[0]:search_roi[0]+search_roi[2], :]
            current_features = optical_flow.update(frame_roi)

            for f in current_features:
                cv2.circle(frame, (int(search_roi[0]+f[0]), int(search_roi[1]+f[1])), 3, (0, 0, 255), -1)

            key = self.ui.preview(
                    frame,
                    frame_num + self.params.start_frame,
                    texte = ["Press 'q' to stop tracking"],
                    boxes = [search_roi],
                )

            if self.ui.was_key_pressed('q') or key == ord('q'):
                status = 'Tracking stopped by user'
                break

        result = optical_flow.get_result()
        result = self.extract_movement(result)

        _, _, _, principalComponents, _ = PPCA(np.transpose(np.array(result, dtype=float)), d=1)
        result = [x[0] for x in principalComponents.tolist()]

        signal = Signal(self.video_info.fps)
        points = signal.get_local_min_max_points(result)
        categorized_points = signal.categorize_points(result, points)

        for k in self.funscripts:
            for bottom_point in categorized_points['max']:
                self.funscripts[k].add_action(
                        0,
                        FFmpegStream.frame_to_millisec(self.get_absolute_framenumber(bottom_point * (1+self.params.skip_frames)), self.video_info.fps)
                    )

            for top_point in categorized_points['min']:
                self.funscripts[k].add_action(
                        100,
                        FFmpegStream.frame_to_millisec(self.get_absolute_framenumber(top_point * (1+self.params.skip_frames)), self.video_info.fps)
                    )

        return status


    def finished(self, status: str, success :bool) -> None:
        """ Process necessary steps to complete the predicted funscript

        Args:
            status (str): a process status/error message
            success (bool): True if funscript was generated else False
        """
        self.ui.close()
        self.funscriptCompleted.emit(self.funscripts, status, success)


    def run(self) -> None:
        try:
            status = self.tracking()
            self.finished(status, True)
        except Exception as ex:
            self.logger.critical("The program crashed due to a fatal error", exc_info=ex)
            self.finished("The program crashed due to a fatal error", False)
