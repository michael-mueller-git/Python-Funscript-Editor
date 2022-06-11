""" Top level process to generate the funscript actions by tracking selected features in the video """

from os import error, read
import cv2
import copy
import time
import math
import funscript_editor.utils.logging as logging
import threading

from dataclasses import dataclass
from PyQt5 import QtCore
from scipy.interpolate import interp1d

from funscript_editor.algorithms.videotracker import StaticVideoTracker
from funscript_editor.data.ffmpegstream import FFmpegStream
from funscript_editor.data.funscript import Funscript
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS, PROJECTION
from funscript_editor.algorithms.scenedetect import SceneDetectFromFile, SceneContentDetector, SceneThresholdDetector
from funscript_editor.algorithms.signal import Signal
from funscript_editor.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters

import multiprocessing as mp
import numpy as np

@dataclass
class FunscriptGeneratorParameter:
    """ Funscript Generator Parameter Dataclass with default values """
    video_path: str
    track_men: bool
    supervised_tracking: bool
    projection: str
    start_frame: int
    end_frame: int = -1 # default is video end (-1)
    number_of_trackers: int = 1
    supervised_tracking_is_exit_condition: bool = True

    # Settings
    points: str = "local_min_max"
    additional_points: str = "none"
    raw_output: bool = SETTINGS["raw_output"]
    max_playback_fps: int = max((0, int(SETTINGS['max_playback_fps'])))
    scene_detector: str = SETTINGS['scene_detector']

    # General Hyperparameter
    skip_frames: int = 1
    top_points_offset: float = 10.0
    bottom_points_offset: float = -10.0


def merge_score(item: list, number_of_trackers: int, return_queue: mp.Queue = None) -> list:
    """ Merge score for given number of trackers

    Note:
        Python multiprocessing methods use a mp.SimpleQueue to pass tasks to the worker processes.
        Everything that goes through the mp.SimpleQueue must be pickable.
        In python functions are only picklable if they are defined at the top-level of a module.

    Args:
        item (list): score for each tracker
        number_of_trackers (int): number of used tracker (pairs)
        return_queue (mp.Queue, optional): return queue to return values via queue

    Returns:
        list: merged score
    """
    if number_of_trackers == 1:
        if return_queue is not None:
            return_queue.put(item[0] if len(item) > 0 else [])
        else:
            return item[0] if len(item) > 0 else []
    else:
        max_frame_number = max([len(item[i]) for i in range(number_of_trackers)])
        arr = np.ma.empty((max_frame_number,number_of_trackers))
        arr.mask = True
        item = np.array(item)
        for tracker_number in range(number_of_trackers):
            arr[:item[tracker_number].shape[0],tracker_number] = item[tracker_number]
        if return_queue is not None:
            return_queue.put(list(filter(None.__ne__, arr.mean(axis=1).tolist())))
        else:
            return list(filter(None.__ne__, arr.mean(axis=1).tolist()))


class FunscriptGeneratorThread(QtCore.QThread):
    """ Funscript Generator Thread

    Args:
        params (FunscriptGeneratorParameter): required parameter for the funscript generator
        funscript (dict): the references to the Funscript where we store the predicted actions
    """

    def __init__(self,
                 params: FunscriptGeneratorParameter,
                 funscripts: dict):
        QtCore.QThread.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.funscripts = funscripts
        self.video_info = FFmpegStream.get_video_info(self.params.video_path)
        self.tracking_points = {}
        self.score = {
                'x': [],
                'y': [],
                'distance': [],
                'roll': []
            }

        self.ui = OpenCV_GUI(OpenCV_GUI_Parameters(
            video_info = self.video_info,
            skip_frames = self.params.skip_frames,
            end_frame_number = self.params.end_frame
            ))


    #: completed event with reference to the funscript with the predicted actions, status message and success flag
    funscriptCompleted = QtCore.pyqtSignal(dict, str, bool)



    def interpolate_bboxes(self, bboxes :dict) -> dict:
        """ Interpolate tracking boxes for skiped frames

        Args:
            bboxes (dict): the new tracking box (x,y,w,h) in dict {Men: {frame_num: tracker_number: {box, ...}}}, Woman: {tracker_number: {box, ...}}}

        Returns:
            dict: interpolated bboxes {'Men': {tracker_number: [(box_frame1),(box_frame2),....], ...} 'Woman': {tracker_number: [(box_frame1),(box_frame2),....], ...}}
        """
        interpolation = 'quadratic'
        if self.params.supervised_tracking and not self.params.supervised_tracking_is_exit_condition:
            # NOTE: wh have to use linea interpolation to get flat lines at when the feature leave the tracking area
            interpolation = 'linear'

        self.logger.info("Use %s interpolation", interpolation)

        interpolated_bboxes = {}
        for tracker_type in bboxes:
            interpolated_bboxes[tracker_type] = {}
            for tracker_number in range(self.params.number_of_trackers):
                interpolated_bboxes[tracker_type][tracker_number] = []
                x, boxes = [], []
                for frame_num in bboxes[tracker_type].keys():
                    if tracker_number in bboxes[tracker_type][frame_num]:
                        x.append(frame_num)
                        boxes.append(bboxes[tracker_type][frame_num][tracker_number])
                if len(boxes) < 2: continue

                # improve border interpolation
                x_head = [x[0]-1]+x+[x[-1]+1]
                boxes = [boxes[0]]+boxes+[boxes[-1]]

                fx0 = interp1d(x_head, [item[0] for item in boxes], kind = interpolation)
                fy0 = interp1d(x_head, [item[1] for item in boxes], kind = interpolation)
                fw  = interp1d(x_head, [item[2] for item in boxes], kind = interpolation)
                fh  = interp1d(x_head, [item[3] for item in boxes], kind = interpolation)

                for i in range(min(x), max(x)+1):
                    interpolated_bboxes[tracker_type][tracker_number].append((float(fx0(i)), float(fy0(i)), float(fw(i)), float(fh(i))))
        return interpolated_bboxes



    def calculate_score(self, bboxes) -> None:
        """ Calculate the score for the predicted tracking boxes

        Note:
            We use x0,y0 from the predicted tracking boxes to create a diff score

        Args:
            bboxes (dict): the preprocessed bboxes
        """
        score = {
                'x':[np.array([]) for _ in range(self.params.number_of_trackers)],
                'y':[np.array([]) for _ in range(self.params.number_of_trackers)],
                'distance': [np.array([]) for _ in range(self.params.number_of_trackers)],
                'roll': [[] for _ in range(self.params.number_of_trackers)]
        }
        self.logger.info("Calculate score for %d Tracker(s)", self.params.number_of_trackers)
        for tracker_number in range(self.params.number_of_trackers):
            woman_center = [self.get_center(item) for item in bboxes['Woman'][tracker_number]]

            if self.params.track_men:
                men_center = [self.get_center(item) for item in bboxes['Men'][tracker_number]]

                score['x'][tracker_number] = np.array([w[0] - m[0] for w, m in zip(woman_center, men_center)])
                score['y'][tracker_number] = np.array([m[1] - w[1] for w, m in zip(woman_center, men_center)])

                score['distance'][tracker_number] = np.array([np.sqrt(np.sum((np.array(m) - np.array(w)) ** 2, axis=0)) \
                        for w, m in zip(woman_center, men_center)])

                for i in range( min(( len(men_center), len(woman_center) )) ):
                    x = woman_center[i][0] - men_center[i][0]
                    y = men_center[i][1] - woman_center[i][1]
                    if x >= 0 and y >= 0:
                        score['roll'][tracker_number].append(np.arctan(np.array(y / max((10e-3, x)))))
                    elif x >= 0 and y < 0:
                        score['roll'][tracker_number].append(-1.0*np.arctan(np.array(y / max((10e-3, x)))))
                    elif x < 0 and y < 0:
                        score['roll'][tracker_number].append(math.pi + -1.0*np.arctan(np.array(y / x)))
                    elif x < 0 and y >= 0:
                        score['roll'][tracker_number].append(math.pi + np.arctan(np.array(y / x)))
                    else:
                        # this should never happen
                        self.logger.error('Calculate score not implement for x=%d, y=%d', x, y)

                # invert because math angle is ccw, also scale to 0- 100
                tmp = score['roll'][tracker_number] # we can not override the list with listcomprehention in python
                score['roll'][tracker_number] = [-100*item/(math.pi) for item in tmp]

            else:
                min_woman_x = min([x[0] for x in woman_center])
                max_woman_y = max([x[1] for x in woman_center])
                score['x'][tracker_number] = np.array([w[0] - min_woman_x for w in woman_center])
                score['y'][tracker_number] = np.array([max_woman_y - w[1] for w in woman_center])

        self.logger.info("Merge Scores")

        pool, queue = {}, {}
        for metric in score.keys():
            queue[metric] = mp.Queue()
            pool[metric] = threading.Thread(target=merge_score, args=(score[metric], self.params.number_of_trackers, queue[metric], ))
            pool[metric].start()

        for metric in score.keys():
            pool[metric].join()
            score[metric] = queue[metric].get()

        for metric in score.keys():
            if metric in self.funscripts.keys() and self.funscripts[metric].is_inverted():
                if metric == 'roll':
                    self.logger.info("%s: Get absolute inverted Score", metric)
                    self.score[metric] = [abs(-1.0 * item) for item in score[metric]]
                else:
                    self.logger.info("%s: Scale Inverted Score to 0 - 100", metric)
                    self.score[metric] = Signal.scale([-1.0 * x for x in score[metric]], 0, 100)
            else:
                if metric == 'roll':
                    self.logger.info("%s: Get absolute Score", metric)
                    self.score[metric] = [abs(item) for item in score[metric]]
                else:
                    self.logger.info("%s: Scale Score to 0 - 100", metric)
                    self.score[metric] = Signal.scale(score[metric], 0, 100)



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
            if self.is_vr_video():
                if 'sbs' in self.params.projection.split('_'):
                    imgMin = imgMin[:, :int(imgMin.shape[1]/2)]
                    imgMax = imgMax[:, :int(imgMax.shape[1]/2)]
                elif 'ou' in self.params.projection.split('_'):
                    imgMin = imgMin[:int(imgMin.shape[0]/2), :]
                    imgMax = imgMax[:int(imgMax.shape[0]/2), :]
                else:
                    self.logger.warning("Unknown VR Projection Type: %s", self.params.projection)

            if PROJECTION[self.params.projection]['parameter']['width'] > 0:
                scale = PROJECTION[self.params.projection]['parameter']['width'] / float(1.75*imgMax.shape[1])
            else:
                scale = PROJECTION[self.params.projection]['parameter']['height'] / float(1.75*imgMax.shape[0])

            imgMin = cv2.resize(imgMin, None, fx=scale, fy=scale)
            imgMax = cv2.resize(imgMax, None, fx=scale, fy=scale)

            min_tracking_points = self.get_tracking_points_by_frame_number(min_frame - self.params.start_frame)
            max_tracking_points = self.get_tracking_points_by_frame_number(max_frame - self.params.start_frame)

            # TODO: draw points to image
            # NOTE: Code below does not work because image do not use the same projection

            # for points in min_tracking_points:
            #     imgMin = OpenCV_GUI.draw_point_to_image(imgMin, points)

            # for points in max_tracking_points:
            #     imgMax = OpenCV_GUI.draw_point_to_image(imgMax, points)

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


    def get_center(self, box: tuple) -> tuple:
        """ Get the cencter point of an box

        Args:
            box (tuple): the predicted bounding box

        Returns:
            tuple (x,y) of the current point
        """
        return ( round(box[0] + box[2]/2), round(box[1] + box[3]/2) )


    def correct_bboxes(self, bboxes: dict, num :int) -> dict:
        """ Delete the latest tracking predictions e.g. to clear bad tracking values

        TODO:
            Delete only from the point where min or max from data before `num` were exceeded or fallen below

        Args:
            bboxes (dict): the raw bboxes
            num (int): number of frames to remove from predicted boxes

        Returns:
            dict: the filtered bboxes
        """
        if self.params.track_men:
            last_relevant_prediction = min((
                max([0]+[k for k in bboxes['Woman'].keys()]),
                max([0]+[k for k in bboxes['Men'].keys()])
            ))
        else:
            last_relevant_prediction = max([0]+[k for k in bboxes['Woman'].keys()])

        if len(bboxes['Woman'].keys()) > 0:
            for k in [k for k in bboxes['Woman'].keys() if k > last_relevant_prediction - num]:
                try: del bboxes['Woman'][k]
                except: pass

        if len(bboxes['Men'].keys()) > 0:
            for k in [k for k in bboxes['Men'].keys() if k > last_relevant_prediction - num]:
                try: del bboxes['Men'][k]
                except: pass

        return bboxes



    def get_target_name(self, person=0) -> str:
        """ Get target name for tracking boxes

        Args:
            person (int): corresponding persion id

        Returns:
            str: name
        """
        if not self.params.track_men:
            return "moving person"
        else:
            return "top/left person" if person == 0 else "bottom/right person"



    def init_trackers(self, first_frame: np.array) -> tuple:
        """ Initialize the trackers

        Args:
            first_frame (np.array): first frame to init the trackers

        Returns:
            tuple: (first_frame, bboxes, tracking_areas_woman, tracking_areas_men, trackers_woman, trackers_men)
        """
        bboxes = {
                'Men': {},
                'Woman': {}
            }

        tracking_areas_woman = [(0,0,0,0) for _ in range(self.params.number_of_trackers)]
        tracking_areas_men = [(0,0,0,0) for _ in range(self.params.number_of_trackers)]
        trackers_woman = [None for _ in range(self.params.number_of_trackers)]
        trackers_men = [None for _ in range(self.params.number_of_trackers)]

        preview_frame = first_frame
        for tracker_number in range(self.params.number_of_trackers):
            bbox_woman = self.ui.bbox_selector(
                    preview_frame,
                    "Select {} Feature #{}".format(self.get_target_name(0), tracker_number+1),
                    add_center = True
                )

            preview_frame = self.ui.draw_box_to_image(
                    preview_frame,
                    bbox_woman,
                    color=(255,0,255)
                )

            if self.params.supervised_tracking:
                while True:
                    tracking_areas_woman[tracker_number] = self.ui.bbox_selector(
                            preview_frame,
                            "Select the Supervised Tracking Area for the {} Feature #{}".format(self.get_target_name(0), tracker_number+1),
                        )

                    if StaticVideoTracker.is_bbox_in_tracking_area(bbox_woman, tracking_areas_woman[tracker_number]):
                        break

                    self.logger.error("Invalid supervised tracking area selected")

                preview_frame = self.ui.draw_box_to_image(
                        preview_frame,
                        tracking_areas_woman[tracker_number],
                        color=(0,255,0)
                    )

                trackers_woman[tracker_number] = StaticVideoTracker(
                        first_frame,
                        bbox_woman,
                        self.video_info.fps,
                        supervised_tracking_area = tracking_areas_woman[tracker_number],
                        supervised_tracking_is_exit_condition=self.params.supervised_tracking_is_exit_condition
                    )
            else:
                trackers_woman[tracker_number] = StaticVideoTracker(
                        first_frame,
                        bbox_woman,
                        self.video_info.fps
                    )

            if tracker_number == 0:
                bboxes['Woman'][1] = { tracker_number: bbox_woman }
            else:
                bboxes['Woman'][1][tracker_number] = bbox_woman

            if self.params.track_men:
                bbox_men = self.ui.bbox_selector(
                        preview_frame,
                        "Select {} Feature #{}".format(self.get_target_name(1), tracker_number+1),
                        add_center = True
                    )
                preview_frame = self.ui.draw_box_to_image(preview_frame, bbox_men, color=(255,0,255))
                if self.params.supervised_tracking:
                    while True:
                        tracking_areas_men[tracker_number] = self.ui.bbox_selector(
                                preview_frame,
                                "Select the Supervised Tracking Area for the {} Feature #{}".format(self.get_target_name(1), tracker_number+1)
                            )

                        if StaticVideoTracker.is_bbox_in_tracking_area(bbox_men, tracking_areas_men[tracker_number]):
                            break

                        self.logger.error("Invalid supervised tracking area selected")

                    preview_frame = self.ui.draw_box_to_image(
                            preview_frame,
                            tracking_areas_men[tracker_number],
                            color=(255,0,255)
                        )

                    trackers_men[tracker_number] = StaticVideoTracker(
                            first_frame,
                            bbox_men,
                            self.video_info.fps,
                            supervised_tracking_area = tracking_areas_men[tracker_number],
                            supervised_tracking_is_exit_condition = self.params.supervised_tracking_is_exit_condition
                        )
                else:
                    trackers_men[tracker_number] = StaticVideoTracker(
                            first_frame,
                            bbox_men,
                            self.video_info.fps
                        )

                if tracker_number == 0:
                    bboxes['Men'][1] = { tracker_number: bbox_men }
                else:
                    bboxes['Men'][1][tracker_number] = bbox_men

        return (first_frame, bboxes, tracking_areas_woman, tracking_areas_men, trackers_woman, trackers_men)


    def tracking(self) -> str:
        """ Tracking function to track the features in the video

        TODO:
            bring back Tracking lost function

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

        (first_frame, bboxes, tracking_areas_woman, tracking_areas_men, trackers_woman, trackers_men) = self.init_trackers(first_frame)

        if self.params.max_playback_fps > (self.params.skip_frames+1):
            cycle_time_in_ms = (float(1000) / float(self.params.max_playback_fps)) * (self.params.skip_frames+1)
        else:
            cycle_time_in_ms = 0

        if self.params.scene_detector.upper() == "CONTENT":
            scene_detector = SceneContentDetector(self.params.start_frame, first_frame, self.params.skip_frames, self.video_info.fps)
        elif self.params.scene_detector.upper() == "THRESHOLD":
            scene_detector = SceneThresholdDetector(self.params.start_frame, self.params.skip_frames, self.video_info.fps)
        else:
            # Fallback is SceneDetectFromFile
            scene_detector = SceneDetectFromFile(self.params.video_path, self.params.start_frame, self.params.skip_frames, self.video_info.fps)

        status = "End of video reached"
        last_frame, frame_num = None, 1 # first frame is init frame
        delete_last_predictions = 0
        bbox_woman = [None for _ in range(self.params.number_of_trackers)]
        bbox_men = [None for _ in range(self.params.number_of_trackers)]
        self.ui.clear_keypress_queue()
        try:
            while video.isOpen():
                cycle_start = time.time()
                frame = video.read()
                frame_num += (self.params.skip_frames+1)

                if frame is None:
                    status = 'Reach a corrupt video frame' if video.isOpen() else 'End of video reached'
                    break

                if self.params.end_frame > 0 and frame_num + self.params.start_frame >= self.params.end_frame:
                    status = "Tracking stop at existing action point"
                    break

                for tracker_number in range(self.params.number_of_trackers):
                    trackers_woman[tracker_number].update(frame)
                    if self.params.track_men: trackers_men[tracker_number].update(frame)
                scene_detector.update(frame)

                if last_frame is not None:
                    # Process data from last step while the next tracking points get predicted.
                    # This should improve the whole processing speed, because the tracker run in a seperate thread
                    boxes_to_draw = []
                    for tracker_number in range(self.params.number_of_trackers):
                        if bbox_woman[tracker_number] is not None:
                            if tracker_number == 0:
                                bboxes['Woman'][frame_num-1] = { tracker_number: bbox_woman[tracker_number] }
                            else:
                                bboxes['Woman'][frame_num-1][tracker_number] = bbox_woman[tracker_number]
                            boxes_to_draw.append(bboxes['Woman'][frame_num-1][tracker_number])
                            if self.params.supervised_tracking:
                                boxes_to_draw.append(tracking_areas_woman[tracker_number])

                        if self.params.track_men and bbox_men[tracker_number] is not None:
                            if tracker_number == 0:
                                bboxes['Men'][frame_num-1] = { tracker_number: bbox_men[tracker_number] }
                            else:
                                bboxes['Men'][frame_num-1][tracker_number] = bbox_men[tracker_number]
                            boxes_to_draw.append(bboxes['Men'][frame_num-1][tracker_number])
                            if self.params.supervised_tracking:
                                boxes_to_draw.append(tracking_areas_men[tracker_number])


                    scene_change_quit_flag = False
                    if scene_detector.is_scene_change(frame_num-1 + self.params.start_frame):
                        self.logger.info("Scene change detected, Pause tracking")
                        key = self.ui.preview(
                                last_frame,
                                frame_num + self.params.start_frame,
                                texte = ["Scene change detected, 'space': continue, 'q': stop"],
                                boxes = boxes_to_draw,
                                beep = True
                            )
                        while True:
                            if self.ui.was_any_accept_key_pressed() or any(key == x for x in [ord(' '), 13]):
                                break

                            if self.ui.was_key_pressed('q') or key == ord('q'):
                                scene_change_quit_flag = True
                                break

                            key = cv2.waitKey(25)

                    if scene_change_quit_flag:
                        status = 'Tracking stopped at scene change'
                        if self.params.scene_detector.upper() == "THRESHOLD":
                            # NOTE: The threshold scene detector has delayed detection
                            delete_last_predictions = int(self.video_info.fps)
                        else:
                            delete_last_predictions = (self.params.skip_frames+1)*2
                        break

                    key = self.ui.preview(
                            last_frame,
                            frame_num + self.params.start_frame,
                            texte = ["Press 'q' to stop tracking"],
                            boxes = boxes_to_draw,
                        )

                    if self.ui.was_key_pressed('q') or key == ord('q'):
                        status = 'Tracking stopped by user'
                        delete_last_predictions = max((1, int((self.ui.get_preview_fps()+1)*0.5*HYPERPARAMETER['user_reaction_time_in_milliseconds']/1000.0)))
                        break

                stop_tracking = False
                for tracker_number in range(self.params.number_of_trackers):
                    (woman_tracker_status, bbox_woman[tracker_number]) = trackers_woman[tracker_number].result()
                    if woman_tracker_status != StaticVideoTracker.Status.OK:
                        status = '{}.1 '.format(tracker_number+1) + woman_tracker_status
                        delete_last_predictions = (self.params.skip_frames+1)*3
                        stop_tracking = True
                        break

                    if self.params.track_men:
                        (men_tracker_status, bbox_men[tracker_number]) = trackers_men[tracker_number].result()
                        if men_tracker_status != StaticVideoTracker.Status.OK:
                            status = '{}.2 '.format(tracker_number+1) + men_tracker_status
                            delete_last_predictions = (self.params.skip_frames+1)*3
                            stop_tracking = True
                            break

                if stop_tracking:
                    break

                last_frame = frame

                if cycle_time_in_ms > 0:
                    wait = cycle_time_in_ms - (time.time() - cycle_start)*float(1000)
                    if wait > 0: time.sleep(wait/float(1000))
        except Exception as ex:
            self.logger.critical("The program crashed due to a fatal error", exc_info=ex)
            return "program crashed"

        if video.isTimeout():
            status = "Reach a corrupt video frame"

        for i in range(self.params.number_of_trackers):
            trackers_woman[i].stop()
            if self.params.track_men:
                trackers_men[i].stop()

        self.ui.show_loading_screen()
        self.logger.info("Raw tracking data: %d Tracking points for %d seconds of the video", \
                len(bboxes["Woman"]), int(len(bboxes["Woman"])*(self.params.skip_frames + 1)/self.video_info.fps))
        video.stop()
        bboxes = self.correct_bboxes(bboxes, delete_last_predictions)
        self.logger.info(status)
        self.logger.info('Interpolate tracking boxes')
        interpolated_bboxes = self.interpolate_bboxes(bboxes)
        self.tracking_points = self.determine_tracking_points(interpolated_bboxes)
        self.calculate_score(interpolated_bboxes)
        return status


    def determine_tracking_points(self, interpolated_bboxes: dict) -> dict:
        """ Determine the final tracking points

        Args:
            interpolate_bboxes (dict): interpolate bboxes from all trackers

        Returns:
            dict: final tracking points
        """
        result = {}
        for tracker_type in interpolated_bboxes.keys():
            result[tracker_type] = {}
            for tracker_number in interpolated_bboxes[tracker_type].keys():
                result[tracker_type][tracker_number] = [self.get_center(item) for item in interpolated_bboxes[tracker_type][tracker_number]]

        return result


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


    def get_absolute_framenumber(self, frame_number: int) -> int:
        """ Get the absoulte frame number

        Args:
            frame_number (int): relative frame number

        Returns:
            int: absolute frame position
        """
        return self.params.start_frame + frame_number


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

        signal = Signal(self.video_info.fps)
        decimate_indexes = signal.decimate(
                self.score[metric],
                base_point_algorithm,
                additional_points_algorithms
        )
        return signal.categorize_points(self.score[metric], decimate_indexes)


    def is_vr_video(self):
        """ Check if current video is set to VR

        Returns:
            bool: true if VR is selected else false
        """
        return 'vr' in self.params.projection.lower().split('_')


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


    def run(self) -> None:
        """ The Funscript Generator Thread Function """
        try:
            if any(metric not in ['x', 'y'] for metric in self.funscripts.keys()):
                self.logger.info('Force 2 person tracking')
                self.params.track_men = True # we need 2 tracking points

            if self.video_info.fps < 31.0 and self.params.skip_frames > 1:
                self.logger.warning("The Video has less than 30 frames per seconds and you have set skip_frames to %d "\
                        + "this can lead to inaccuracies when predicting the changepoint positions! (consider to set skip_frames to 0 or 1)" \
                        , self.params.skip_frames)

            if self.params.raw_output:
                self.logger.warning("Raw output is enabled!")

            status = self.tracking()

            for metric in self.funscripts.keys():
                if len(self.score[metric]) >= HYPERPARAMETER['min_frames']:
                    self.scale_score(status, metric=metric)

            if all(len(self.score[metric]) < HYPERPARAMETER['min_frames'] for metric in self.funscripts.keys()):
                self.finished(
                        status + ' -> Tracking time insufficient ({}/{} Frames)'.format(
                            min([len(self.score[metric]) for metric in self.funscripts.keys()]),
                            HYPERPARAMETER['min_frames']
                        ), False
                )
                return

            for metric in self.funscripts.keys():
                idx_dict = self.determine_change_points(metric)
                self.create_funscript(idx_dict, metric)

            self.finished(status, True)
        except Exception as ex:
            self.logger.critical("The program crashed due to a fatal error", exc_info=ex)
            self.finished("The program crashed due to a fatal error", False)
