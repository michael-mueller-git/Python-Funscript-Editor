from os import path
import cv2
import time
import json
import copy
import logging

import numpy as np

from funscript_editor.data.ffmpegstream import FFmpegStream
from dataclasses import dataclass
from sklearn.decomposition import PCA
from PyQt5 import QtCore
from funscript_editor.algorithms.signal import Signal
from funscript_editor.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters

@dataclass
class OpticalFlowFunscriptGeneratorParameter:
    """ Funscript Generator Parameter Dataclass with default values """
    video_path: str
    projection: str
    start_frame: int
    end_frame: int = -1 # default is video end (-1)
    skip_frames: int = 0
    min_trajectory_len: int = 60
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

        def __init__(self, min_trajectory_len, feature_detect_interval):
            self.min_trajectory_len = min_trajectory_len
            self.feature_detect_interval = feature_detect_interval
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
                            self.result.append({'end': self.frame_idx, 'trajectory': trajectory})
                        continue
                    trajectory.append((x, y))
                    new_trajectories.append(trajectory)

                self.trajectories = new_trajectories


            if self.frame_idx % self.feature_detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        if any(abs(t[-1][0] - x) < 3 and abs(t[-1][1] - y) < 3 for t in self.trajectories):
                            continue

                        self.trajectories.append([(x, y)])


            self.frame_idx += 1
            self.prev_frame_gray = frame_gray

            return [t[-1] for t in self.trajectories]


        def get_result(self):
            result = copy.deepcopy(self.result)
            for trajectory in self.trajectories:
                if len (trajectory) > self.min_trajectory_len:
                    result.append({'end': self.frame_idx, 'trajectory': trajectory})

            return { 'meta': { 'last_idx': self.frame_idx }, 'data': result }


    def extract_movement(self, optical_flow_result, metric_idx = 1):
        result = []
        for r in optical_flow_result['data']:
            zero_before = r['end'] - len(r['trajectory'])
            zero_after = optical_flow_result['meta']['last_idx'] - r['end']
            trajectory_min = min([item[metric_idx] for item in r['trajectory']])
            y = [0 for _ in range(zero_before)] + [(r['trajectory'][i][metric_idx] - trajectory_min)**2 for i in range(len(r['trajectory']))] + [0 for _ in range(zero_after)]
            if max(y) - min(y) > self.params.movement_filter:
                result.append(y)

        return result


    def get_absolute_framenumber(self, frame_number: int) -> int:
        """ Get the absoulte frame number

        Args:
            frame_number (int): relative frame number

        Returns:
            int: absolute frame position
        """
        return self.params.start_frame + frame_number


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

        roi = self.ui.bbox_selector(
            first_frame,
            "Select ROI",
        )

        optical_flow = OpticalFlowFunscriptGeneratorThread.OpticalFlowPyrLK(
                min_trajectory_len = self.params.min_trajectory_len,
                feature_detect_interval = self.params.feature_detect_interval
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

            frame_roi = frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2], :]
            current_features = optical_flow.update(frame_roi)

            for f in current_features:
                cv2.circle(frame, (int(roi[0]+f[0]), int(roi[1]+f[1])), 3, (0, 0, 255), -1)

            key = self.ui.preview(
                    frame,
                    frame_num + self.params.start_frame,
                    texte = ["Press 'q' to stop tracking"],
                    boxes = [roi],
                )

            if self.ui.was_key_pressed('q') or key == ord('q'):
                status = 'Tracking stopped by user'
                break

        result = optical_flow.get_result()
        result = self.extract_movement(result)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(np.transpose(np.array(result)))
        result = np.transpose(np.array(principalComponents))

        result = np.array(result[0]) - np.array(result[1])

        signal = Signal(self.video_info.fps)
        points = signal.get_local_min_max_points(result)

        val = 0
        for k in self.funscripts:
            for p in points:
                self.funscripts[k].add_action(
                        val,
                        FFmpegStream.frame_to_millisec(self.get_absolute_framenumber(p * (1+self.params.skip_frames)), self.video_info.fps)
                    )
                val = 0 if val  != 0 else 100

        return status


    def finished(self, status: str, success :bool) -> None:
        """ Process necessary steps to complete the predicted funscript

        Args:
            status (str): a process status/error message
            success (bool): True if funscript was generated else False
        """
        for metric in self.funscripts.keys():
            # we use this flag internaly to determine if we have to invert the score
            # ensure not to publish the invertion with our generated funscript
            # in this case we will invert our result again, what we dont want
            self.funscripts[metric].data["inverted"] = False
        self.ui.close()
        self.funscriptCompleted.emit(self.funscripts, status, success)


    def run(self) -> None:
        try:
            status = self.tracking()
            self.finished(status, True)
        except Exception as ex:
            self.logger.critical("The program crashed due to a fatal error", exc_info=ex)
            self.finished("The program crashed due to a fatal error", False)
