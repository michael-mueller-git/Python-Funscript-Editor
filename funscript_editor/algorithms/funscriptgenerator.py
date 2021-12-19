""" Top level process to generate the funscript actions by tracking selected features in the video """

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
    metric: str
    projection: str
    invert: bool
    start_frame: int
    end_frame: int = -1 # default is video end (-1)
    number_of_trackers: int = 1

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
        funscript (Funscript): the reference to the Funscript where we store the predicted actions
    """

    def __init__(self,
                 params: FunscriptGeneratorParameter,
                 funscript: Funscript):
        QtCore.QThread.__init__(self)
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.funscript = funscript
        self.video_info = FFmpegStream.get_video_info(self.params.video_path)
        self.tracking_fps = []
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
    funscriptCompleted = QtCore.pyqtSignal(object, str, bool)



    def interpolate_bboxes(self, bboxes :dict) -> dict:
        """ Interpolate tracking boxes for skiped frames

        Args:
            bboxes (dict): the new tracking box (x,y,w,h) in dict {Men: {frame_num: tracker_number: {box, ...}}}, Woman: {tracker_number: {box, ...}}}

        Returns:
            dict: interpolated bboxes {'Men': {tracker_number: [(box_frame1),(box_frame2),....], ...} 'Woman': {tracker_number: [(box_frame1),(box_frame2),....], ...}}
        """
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

                fx0 = interp1d(x_head, [item[0] for item in boxes], kind = 'quadratic')
                fy0 = interp1d(x_head, [item[1] for item in boxes], kind = 'quadratic')
                fw  = interp1d(x_head, [item[2] for item in boxes], kind = 'quadratic')
                fh  = interp1d(x_head, [item[3] for item in boxes], kind = 'quadratic')

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

                score['x'][tracker_number] = np.array([w[0] - m[0] for w, m in zip(bboxes['Woman'][tracker_number], bboxes['Men'][tracker_number])])
                score['y'][tracker_number] = np.array([m[1] - w[1] for w, m in zip(bboxes['Woman'][tracker_number], bboxes['Men'][tracker_number])])

                score['distance'][tracker_number] = np.array([np.sqrt(np.sum((np.array(m) - np.array(w)) ** 2, axis=0)) \
                        for w, m in zip(woman_center, men_center)])

                for i in range( min(( len(men_center), len(woman_center) )) ):
                    x = bboxes['Woman'][tracker_number][i][0] - bboxes['Men'][tracker_number][i][0]
                    y = bboxes['Men'][tracker_number][i][1] - bboxes['Woman'][tracker_number][i][1]
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

                # invert because math angle is ccw
                inverted_roll = copy.deepcopy(score['roll'][tracker_number])
                score['roll'][tracker_number] = -1*np.array(inverted_roll)


            else:
                score['x'][tracker_number] = np.array([w[0] - min([x[0] for x in bboxes['Woman'][tracker_number]]) for w in bboxes['Woman'][tracker_number]])
                score['y'][tracker_number] = np.array([max([x[1] for x in bboxes['Woman'][tracker_number]]) - w[1] for w in bboxes['Woman'][tracker_number]])

        self.logger.info("Merge Scores")

        pool, queue = {}, {}
        for metric in score.keys():
            queue[metric] = mp.Queue()
            pool[metric] = threading.Thread(target=merge_score, args=(score[metric], self.params.number_of_trackers, queue[metric], ))
            pool[metric].start()

        for metric in score.keys():
            pool[metric].join()
            score[metric] = queue[metric].get()

        if self.params.invert:
            self.logger.info("Scale Inverted Score to 0 - 100")
            for metric in score.keys():
                self.score[metric] = Signal.scale(-1.0*np.array(score[metric]), 0, 100)
        else:
            self.logger.info("Scale Score to 0 - 100")
            for metric in score.keys():
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

        cap = cv2.VideoCapture(self.params.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
        success_min, imgMin = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_frame)
        success_max, imgMax = cap.read()
        cap.release()

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

            (desired_min, desired_max) = self.ui.min_max_selector(
                    image_min = imgMin,
                    image_max = imgMax,
                    info = status,
                    title_min = metric + " Minimum",
                    title_max = metric + " Maximum"
                )
        else:
            self.logger.warning("Determine min and max failed")
            desired_min = 0
            desired_max = 99

        self.logger.info("Scale score %s to user input", metric)
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

        mapping = {
            'y': {
                0: 'top person',
                1: 'bottom person'
                },
            'x': {
                0: 'left person',
                1: 'right person'
            },
            'distance': {
                0: 'top/left person',
                1: 'bottom/right person'
            },
            'roll': {
                0: 'top/left person',
                1: 'bottom/right person'
            }
        }

        return mapping[self.params.metric][person]



    def init_trackers(self, ffmpeg_stream: FFmpegStream) -> tuple:
        """ Initialize the trackers

        Args:
            ffmpeg_stream (FFmpegStream): The ffmpeg stream

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

        first_frame = ffmpeg_stream.read()
        preview_frame = first_frame
        for tracker_number in range(self.params.number_of_trackers):
            bbox_woman = self.ui.bbox_selector(preview_frame, "Select {} Feature #{}".format(self.get_target_name(0), tracker_number+1))
            preview_frame = self.ui.draw_box_to_image(preview_frame, bbox_woman, color=(255,0,255))
            if self.params.supervised_tracking:
                while True:
                    tracking_areas_woman[tracker_number] = self.ui.bbox_selector(preview_frame, "Select the Supervised Tracking Area for the {} Feature #{}".format(self.get_target_name(0), tracker_number+1))
                    if StaticVideoTracker.is_bbox_in_tracking_area(bbox_woman, tracking_areas_woman[tracker_number]): break
                    self.logger.error("Invalid supervised tracking area selected")
                preview_frame = self.ui.draw_box_to_image(preview_frame, tracking_areas_woman[tracker_number], color=(0,255,0))
                trackers_woman[tracker_number] = StaticVideoTracker(first_frame, bbox_woman, self.video_info.fps, supervised_tracking_area = tracking_areas_woman[tracker_number])
            else:
                trackers_woman[tracker_number] = StaticVideoTracker(first_frame, bbox_woman, self.video_info.fps)

            if tracker_number == 0:
                bboxes['Woman'][1] = { tracker_number: bbox_woman }
            else:
                bboxes['Woman'][1][tracker_number] = bbox_woman

            if self.params.track_men:
                bbox_men = self.ui.bbox_selector(preview_frame, "Select {} Feature #{}".format(self.get_target_name(1), tracker_number+1))
                preview_frame = self.ui.draw_box_to_image(preview_frame, bbox_men, color=(255,0,255))
                if self.params.supervised_tracking:
                    while True:
                        tracking_areas_men[tracker_number] = self.ui.bbox_selector(preview_frame, "Select the Supervised Tracking Area for the {} Feature #{}".format(self.get_target_name(1), tracker_number+1))
                        if StaticVideoTracker.is_bbox_in_tracking_area(bbox_men, tracking_areas_men[tracker_number]): break
                        self.logger.error("Invalid supervised tracking area selected")
                    preview_frame = self.ui.draw_box_to_image(preview_frame, tracking_areas_men[tracker_number], color=(255,0,255))
                    trackers_men[tracker_number] = StaticVideoTracker(first_frame, bbox_men, self.video_info.fps, supervised_tracking_area = tracking_areas_men[tracker_number])
                else:
                    trackers_men[tracker_number] = StaticVideoTracker(first_frame, bbox_men, self.video_info.fps)

                if tracker_number == 0:
                    bboxes['Men'][1] = { tracker_number: bbox_men }
                else:
                    bboxes['Men'][1][tracker_number] = bbox_men

        return (first_frame, bboxes, tracking_areas_woman, tracking_areas_men, trackers_woman, trackers_men)


    def tracking(self) -> str:
        """ Tracking function to track the features in the video

        TODO:
            Tracking lost for multiple tracker

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

        (first_frame, bboxes, tracking_areas_woman, tracking_areas_men, trackers_woman, trackers_men) = self.init_trackers(video)

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
                                texte = ["Scene change detected, Press 'space' to continue tracking or press 'q' to finalize tracking"],
                                boxes = boxes_to_draw,
                                beep = True
                            )
                        while True:
                            if self.ui.was_space_pressed() or key == ord(' '):
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
                            texte = ["Press 'q' if the tracking point shifts or a video cut occured"],
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
                        status = 'Woman ' + woman_tracker_status
                        delete_last_predictions = (self.params.skip_frames+1)
                        stop_tracking = True
                        break

                    if self.params.track_men:
                        (men_tracker_status, bbox_men[tracker_number]) = trackers_men[tracker_number].result()
                        if men_tracker_status != StaticVideoTracker.Status.OK:
                            status = 'Men ' + men_tracker_status
                            delete_last_predictions = (self.params.skip_frames+1)
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


    def finished(self, status: str, success :bool) -> None:
        """ Process necessary steps to complete the predicted funscript

        Args:
            status (str): a process status/error message
            success (bool): True if funscript was generated else False
        """
        self.ui.close()
        self.funscriptCompleted.emit(self.funscript, status, success)


    def apply_shift(self, frame_number: int, metric: str, position: str) -> int:
        """ Apply shift to predicted frame positions

        Args:
            frame_number (int): relative frame number
            metric (str): metric to apply the shift
            position (str): keyword ['max', 'min', 'None']

        Returns:
            int: real frame position
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


    def create_funscript(self, idx_dict: dict) -> None:
        """ Generate the Funscript

        Args:
            idx_dict (dict): dictionary with all local max and min points in score
                             {'min':[idx1, idx2, ...], 'max':[idx1, idx2, ...]}
        """
        if self.params.raw_output:
            output_score = copy.deepcopy(self.score[self.params.metric])
            for idx in range(len(output_score)):
                self.funscript.add_action(
                        output_score[idx],
                        FFmpegStream.frame_to_millisec(self.apply_shift(idx, self.params.metric, 'None'), self.video_info.fps)
                    )

        else:
            output_score = self.get_score_with_offset(idx_dict, self.params.metric)

            for idx in idx_dict['min']:
                self.funscript.add_action(
                        round(output_score[idx]),
                        FFmpegStream.frame_to_millisec(self.apply_shift(idx, self.params.metric, 'min'), self.video_info.fps)
                    )

            for idx in idx_dict['max']:
                self.funscript.add_action(
                        round(output_score[idx]),
                        FFmpegStream.frame_to_millisec(self.apply_shift(idx, self.params.metric, 'max'), self.video_info.fps)
                    )


    def run(self) -> None:
        """ The Funscript Generator Thread Function """
        try:
            if self.params.metric not in ['x', 'y']:
                self.params.track_men = True # we need 2 tracking points

            if self.video_info.fps < 31.0 and self.params.skip_frames > 1:
                self.logger.warning("The Video has less than 30 frames per seconds and you have set skip_frames to %d "\
                        + "this can lead to inaccuracies when predicting the changepoint positions! (consider to set skip_frames to 0 or 1)" \
                        , self.params.skip_frames)

            if self.params.raw_output:
                self.logger.warning("Raw output is enabled!")

            status = self.tracking()

            if len(self.score[self.params.metric]) >= HYPERPARAMETER['min_frames']:
                self.scale_score(status, metric=self.params.metric)

            if len(self.score[self.params.metric]) < HYPERPARAMETER['min_frames']:
                self.finished(
                        status + ' -> Tracking time insufficient ({}/{} Frames)'.format(
                            len(self.score[self.params.metric]),
                            HYPERPARAMETER['min_frames']
                        ), False
                )
                return

            idx_dict = self.determine_change_points(self.params.metric)

            self.create_funscript(idx_dict)
            self.finished(status, True)
        except Exception as ex:
            self.logger.critical("The program crashed due to a fatal error", exc_info=ex)
            self.finished("The program crashed due to a fatal error", False)
