""" Top level process to track selected features in the video """

import cv2
import time
import math
import funscript_editor.utils.logging as logging
import threading

from dataclasses import dataclass
from PyQt5 import QtCore
from scipy.interpolate import interp1d

from funscript_editor.algorithms.videotracker import StaticVideoTracker
from funscript_editor.data.ffmpegstream import FFmpegStream
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS
from funscript_editor.algorithms.scenedetect import SceneDetectFromFile, SceneContentDetector, SceneThresholdDetector
from funscript_editor.algorithms.signal import Signal
from funscript_editor.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters

import multiprocessing as mp
import numpy as np

@dataclass
class TrackingManagerParameter:
    """ Funscript Generator Parameter Dataclass with default values """
    video_path: str
    projection: str
    start_frame: int
    track_men: bool
    supervised_tracking: bool
    tracking_metrics: dict # key metric, value { inverted: bool }
    end_frame: int = -1 # default is video end (-1)
    number_of_trackers: int = 1
    supervised_tracking_is_exit_condition: bool = True
    skip_frames: int = 1
    max_playback_fps: int = max((0, int(SETTINGS['max_playback_fps'])))
    scene_detector: str = SETTINGS['scene_detector']


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


class TrackingManagerThread(QtCore.QThread):
    """ Tracking Manager Thread

    Args:
        params (TrackingManagerParameter): required parameter for the tracking manager
    """

    def __init__(self,
                 video_info,
                 params: TrackingManagerParameter):
        QtCore.QThread.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.video_info = video_info
        self.tracking_points = {}
        self.projection_config = {'name': 'None', 'video_filter': ''}
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


    #: completed event with tracking scores, projection config, raw tracking points, status message and success flag
    trackingCompleted = QtCore.pyqtSignal(dict, dict, dict, str, bool)



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


    @staticmethod
    def clamp(val: float, min_value: float, max_value: float) -> float:
        """ limit a value in a given range

        Args:
            val (float): input value
            min_value (float): min allowed value
            max_value (float): max allowed value

        Returns:
            float: clamp value
        """
        return max(min(val, max_value), min_value)


    def get_dick_pos(self, max_distance_frame_num: int) -> dict:
        """ Get Start and End points of the dick

        Args:
            max_distance_frame_num (int): absolute frame number with max tracker distance

        Returns:
            dict: dick points
        """
        max_distance_frame = FFmpegStream.get_frame(self.params.video_path, max_distance_frame_num)
        max_distance_frame = FFmpegStream.get_projection(max_distance_frame, self.projection_config)
        frame_h, frame_w = max_distance_frame.shape[:2]
        center_line = self.ui.line_selector(max_distance_frame, "draw line on center of dick")

        # last idx: 0 = x, 1 = y
        dick_pos = { 'w': center_line[1], 'm': center_line[0] } \
                if center_line[0][1] > center_line[1][1] \
                else { 'w': center_line[0], 'm': center_line[1] }

        # TODO: dividor is an hyperparameter
        dx = (dick_pos['m'][0] - dick_pos['w'][0]) / 2.5
        dy = (dick_pos['m'][1] - dick_pos['w'][1]) / 2.5
        return {
                'w': (self.clamp(dick_pos['w'][0] - dx, 0, frame_w-1), self.clamp(dick_pos['w'][1] - dy, 0, frame_h-1)),
                'm': (self.clamp(dick_pos['m'][0] + dx, 0, frame_w-1), self.clamp(dick_pos['m'][1] + dy, 0, frame_h-1))
            }


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

        dick_pos = None
        tracking_points_with_offset = {}
        for tracker_type in bboxes.keys():
            tracking_points_with_offset[tracker_type] = {}

        self.logger.info("Calculate score for %d Tracker(s)", self.params.number_of_trackers)
        for tracker_number in range(self.params.number_of_trackers):
            woman_center = [self.get_center(item) for item in bboxes['Woman'][tracker_number]]

            if self.params.track_men:
                men_center = [self.get_center(item) for item in bboxes['Men'][tracker_number]]

                if len(men_center) == 0 or len(woman_center) == 0:
                    score['x'][tracker_number] = []
                    score['y'][tracker_number] = []
                    score['distance'][tracker_number] = []
                    score['roll'][tracker_number] = []
                    continue

                score['x'][tracker_number] = np.array([w[0] - m[0] for w, m in zip(woman_center, men_center)])
                score['y'][tracker_number] = np.array([m[1] - w[1] for w, m in zip(woman_center, men_center)])

                score['distance'][tracker_number] = np.array([np.sqrt(np.sum((np.array(m) - np.array(w)) ** 2, axis=0)) \
                        for w, m in zip(woman_center, men_center)])

                if dick_pos is None:
                    max_distance_frame_num = np.argmax(np.array([abs(x) for x in score['distance'][tracker_number]])) + self.params.start_frame
                    dick_pos = self.get_dick_pos(max_distance_frame_num)
                    dick_pos['idx'] = max_distance_frame_num - self.params.start_frame

                roll_woman_offset = (dick_pos['w'][0] - woman_center[dick_pos['idx']][0], dick_pos['w'][1] - woman_center[dick_pos['idx']][1])
                roll_men_offset = (dick_pos['m'][0] - men_center[dick_pos['idx']][0], dick_pos['m'][1] - men_center[dick_pos['idx']][1])
                self.logger.info('use roll offset w = %s, m = %s', str(roll_woman_offset), str(roll_men_offset))

                tracking_points_with_offset['Woman'][tracker_number] = [
                        (
                            int(woman_center[i][0] + roll_woman_offset[0]),
                            int(woman_center[i][1] + roll_woman_offset[1])
                        ) for i in range( min(( len(men_center), len(woman_center) )) )
                    ]

                tracking_points_with_offset['Men'][tracker_number] = [
                        (
                            int(men_center[i][0] + roll_men_offset[0]),
                            int(men_center[i][1] + roll_men_offset[1])
                        ) for i in range( min(( len(men_center), len(woman_center) )) )
                    ]

                for i in range( min(( len(men_center), len(woman_center) )) ):
                    x = woman_center[i][0] + roll_woman_offset[0] - (men_center[i][0] + roll_men_offset[0])
                    y = men_center[i][1] + roll_men_offset[1] - (woman_center[i][1] + roll_woman_offset[1])
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
                score['roll'][tracker_number] = [-100*item/(math.pi) + 100 for item in tmp]

            else:
                if len(woman_center) == 0:
                    score['x'][tracker_number] = []
                    score['y'][tracker_number] = []
                    continue

                min_woman_x = min([x[0] for x in woman_center])
                max_woman_y = max([x[1] for x in woman_center])
                score['x'][tracker_number] = np.array([w[0] - min_woman_x for w in woman_center])
                score['y'][tracker_number] = np.array([max_woman_y - w[1] for w in woman_center])

                tracking_points_with_offset['Woman'][tracker_number] = woman_center


        self.tracking_points = tracking_points_with_offset # save so we can use them later for visu

        self.logger.info("Merge Scores")

        pool, queue = {}, {}
        for metric in score.keys():
            queue[metric] = mp.Queue()
            pool[metric] = threading.Thread(target=merge_score, args=(score[metric], self.params.number_of_trackers, queue[metric], ))
            pool[metric].start()

        for metric in score.keys():
            pool[metric].join()
            score[metric] = queue[metric].get()

        self.logger.info("tracking_metrics: %s", self.params.tracking_metrics)
        for metric in score.keys():
            if metric in self.params.tracking_metrics and "inverted" in self.params.tracking_metrics[metric] and self.params.tracking_metrics[metric]["inverted"]:
                if metric == 'roll':
                    self.logger.info("%s: Get absolute inverted Score", metric)
                    self.score[metric] = [abs(-1.0*item + 100) for item in score[metric]]
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
        if first_frame is None or first_frame.shape[0] == 0 or first_frame.shape[1] == 0:
            return "Could not extract first frame, please re-encode your video"

        self.projection_config = self.ui.get_video_projection_config(first_frame, self.params.projection)
        print("Use projection config:", self.projection_config)

        video = FFmpegStream(
                video_path = self.params.video_path,
                config = self.projection_config,
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
                    single_boxes_to_draw = []
                    all_connected_boxes_to_draw = []
                    for tracker_number in range(self.params.number_of_trackers):
                        connected_boxes_to_draw = []
                        if bbox_woman[tracker_number] is not None:
                            if tracker_number == 0:
                                bboxes['Woman'][frame_num-1] = { tracker_number: bbox_woman[tracker_number] }
                            else:
                                bboxes['Woman'][frame_num-1][tracker_number] = bbox_woman[tracker_number]
                            connected_boxes_to_draw.append(bboxes['Woman'][frame_num-1][tracker_number])
                            if self.params.supervised_tracking:
                                single_boxes_to_draw.append(tracking_areas_woman[tracker_number])

                        if self.params.track_men and bbox_men[tracker_number] is not None:
                            if tracker_number == 0:
                                bboxes['Men'][frame_num-1] = { tracker_number: bbox_men[tracker_number] }
                            else:
                                bboxes['Men'][frame_num-1][tracker_number] = bbox_men[tracker_number]
                            connected_boxes_to_draw.append(bboxes['Men'][frame_num-1][tracker_number])
                            if self.params.supervised_tracking:
                                single_boxes_to_draw.append(tracking_areas_men[tracker_number])

                        if len(connected_boxes_to_draw) > 0:
                            all_connected_boxes_to_draw.append(connected_boxes_to_draw)

                    for box in single_boxes_to_draw:
                        all_connected_boxes_to_draw.append([box])

                    scene_change_quit_flag = False
                    if scene_detector.is_scene_change(frame_num-1 + self.params.start_frame):
                        self.logger.info("Scene change detected, Pause tracking")
                        key = self.ui.preview(
                                last_frame,
                                frame_num + self.params.start_frame,
                                texte = ["Scene change detected, 'space': continue, 'q': stop"],
                                boxes = all_connected_boxes_to_draw,
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
                            boxes = all_connected_boxes_to_draw,
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
        self.calculate_score(interpolated_bboxes)
        return status



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
        """ Process necessary steps to complete the predicted tracking score

        Args:
            status (str): a process status/error message
            success (bool): True if tracking was successful else False
        """
        self.ui.close()
        self.trackingCompleted.emit(self.score, self.projection_config, self.tracking_points, status, success)


    def get_absolute_framenumber(self, frame_number: int) -> int:
        """ Get the absoulte frame number

        Args:
            frame_number (int): relative frame number

        Returns:
            int: absolute frame position
        """
        return self.params.start_frame + frame_number


    def is_vr_video(self):
        """ Check if current video is set to VR

        Returns:
            bool: true if VR is selected else false
        """
        return 'vr' in self.params.projection.lower().split('_')


    def run(self) -> None:
        """ The Tracking Manager Thread Function """
        try:
            if any(metric not in ['x', 'y'] for metric in self.params.tracking_metrics):
                self.logger.info('Force 2 person tracking')
                self.params.track_men = True # we need 2 tracking points

            if self.video_info.fps < 31.0 and self.params.skip_frames > 1:
                self.logger.warning("The Video has less than 30 frames per seconds and you have set skip_frames to %d "\
                        + "this can lead to inaccuracies when predicting the changepoint positions! (consider to set skip_frames to 0 or 1)" \
                        , self.params.skip_frames)

            status = self.tracking()

            if all(len(self.score[metric]) < HYPERPARAMETER['min_frames'] for metric in self.params.tracking_metrics):
                self.finished(
                        status + ' -> Tracking time insufficient ({}/{} Frames)'.format(
                            min([len(self.score[metric]) for metric in self.params.tracking_metrics]),
                            HYPERPARAMETER['min_frames']
                        ), False
                )
                return

            self.finished(status, True)
        except Exception as ex:
            self.logger.critical("The program crashed in TrackingManager due to a fatal error", exc_info=ex)
            self.finished("The program crashed in TrackingManager due to a fatal error", False)
