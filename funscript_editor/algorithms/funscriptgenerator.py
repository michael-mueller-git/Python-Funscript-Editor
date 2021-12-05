""" Top level process to generate the funscript actions by tracking selected features in the video """

import cv2
import os
import copy
import time
import math
import json
import logging
import platform
import threading
from numpy.core.fromnumeric import take
from numpy.lib.function_base import append
from numpy.ma.core import array

from playsound import playsound
from screeninfo import get_monitors
from queue import Queue
from pynput.keyboard import Key, Listener
from dataclasses import dataclass
from PyQt5 import QtCore
from matplotlib.figure import Figure
from datetime import datetime
from scipy.interpolate import interp1d

from funscript_editor.algorithms.videotracker import StaticVideoTracker
from funscript_editor.data.ffmpegstream import FFmpegStream
from funscript_editor.data.funscript import Funscript
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS, PROJECTION, NOTIFICATION_SOUND_FILE
from funscript_editor.utils.logging import get_logfiles_paths
from funscript_editor.definitions import SETTINGS_CONFIG_FILE, HYPERPARAMETER_CONFIG_FILE
from funscript_editor.algorithms.scenedetect import SceneDetectFromFile, SceneContentDetector, SceneThresholdDetector
from funscript_editor.algorithms.signal import Signal

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
    use_zoom: bool = SETTINGS['use_zoom']
    zoom_factor: float = max((1.0, float(SETTINGS['zoom_factor'])))
    preview_scaling: float = float(SETTINGS['preview_scaling'])
    tracking_lost_time: int = max((0, SETTINGS['tracking_lost_time']))
    scene_detector: str = SETTINGS['scene_detector']

    # General Hyperparameter
    skip_frames: int = max((0, int(HYPERPARAMETER['skip_frames'])))

    # VR Movement in y Direction
    shift_bottom_points: int = int(HYPERPARAMETER['shift_bottom_points'])
    shift_top_points: int = int(HYPERPARAMETER['shift_top_points'])
    bottom_points_offset: float = float(HYPERPARAMETER['bottom_points_offset'])
    top_points_offset: float = float(HYPERPARAMETER['top_points_offset'])
    bottom_threshold: float = float(HYPERPARAMETER['bottom_threshold'])
    top_threshold: float = float(HYPERPARAMETER['top_threshold'])

    # All other predicted Movements
    shift_min_points: int = int(HYPERPARAMETER['shift_min_points'])
    shift_max_points: int = int(HYPERPARAMETER['shift_max_points'])
    min_points_offset: float = float(HYPERPARAMETER['min_points_offset'])
    max_points_offset: float = float(HYPERPARAMETER['max_points_offset'])
    min_threshold: float = float(HYPERPARAMETER['min_threshold'])
    max_threshold: float = float(HYPERPARAMETER['max_threshold'])


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
        self.params = params
        self.funscript = funscript
        self.video_info = FFmpegStream.get_video_info(self.params.video_path)
        self.timer = cv2.getTickCount()

        # XXX destroyWindow(...) sems not to delete the trackbar. Workaround: we give the window each time a unique name
        self.window_name = "Funscript Generator ({})".format(datetime.now().strftime("%H:%M:%S"))

        self.keypress_queue = Queue(maxsize=32)
        self.x_text_start = 50
        self.font_size = 0.6
        self.tracking_fps = []
        self.score = {
                'x': [],
                'y': [],
                'distance': [],
                'roll': []
            }


    #: completed event with reference to the funscript with the predicted actions, status message and success flag
    funscriptCompleted = QtCore.pyqtSignal(object, str, bool)

    logger = logging.getLogger(__name__)


    def determine_preview_scaling(self, frame_width, frame_height) -> None:
        """ Determine the scaling for current monitor setup

        Args:
            frame_width (int): target frame width
            frame_height (int): target frame height
        """
        scale = []
        try:
            for monitor in get_monitors():
                if monitor.width > monitor.height:
                    scale.append( min((monitor.width / float(frame_width), monitor.height / float(frame_height) )) )
        except: pass

        if len(scale) == 0:
            self.logger.error("Monitor resolution info not found")
        else:
            # asume we use the largest monitor for scipting
            self.params.preview_scaling = float(SETTINGS['preview_scaling']) * max(scale)


    def draw_box(self, img: np.ndarray, bbox: tuple, color: tuple = (255, 0, 255)) -> np.ndarray:
        """ Draw an tracking box on the image/frame

        Args:
            img (np.ndarray): opencv image
            bbox (tuple): tracking box with (x,y,w,h)
            color (tuple): RGB color values for the box

        Returns:
            np.ndarray: opencv image with annotated tracking box
        """
        annotated_img = img.copy()
        cv2.rectangle(annotated_img, (bbox[0], bbox[1]), ((bbox[0]+bbox[2]), (bbox[1]+bbox[3])), color, 3, 1)
        return annotated_img


    def draw_fps(self, img: np.ndarray) -> np.ndarray:
        """ Draw processing FPS on the image/frame

        Args:
            img (np.ndarray): opencv image

        Returns:
            np.ndarray: opencv image with FPS Text
        """
        SMOOTH_FACTOR = 100
        annotated_img = img.copy()
        self.tracking_fps.append((self.params.skip_frames+1)*cv2.getTickFrequency()/(cv2.getTickCount()-self.timer))
        fps = np.mean((self.tracking_fps[-SMOOTH_FACTOR:] if len(self.tracking_fps) < SMOOTH_FACTOR else self.tracking_fps))
        cv2.putText(annotated_img, str(int(fps)) + ' fps', (self.x_text_start, 50),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0,0,255), 2)
        self.timer = cv2.getTickCount()
        return annotated_img


    def draw_time(self, img: np.ndarray, frame_num: int) -> np.ndarray:
        """ Draw Time on the image/frame

        Args:
            img (np.ndarray): opencv image
            img (int): current absolute frame number

        Returns:
            np.ndarray: opencv image with Time Text
        """
        annotated_img = img.copy()
        current_timestamp = FFmpegStream.frame_to_timestamp(frame_num, self.video_info.fps)
        current_timestamp = ''.join(current_timestamp[:-4])

        if self.params.end_frame < 1:
            end_timestamp = FFmpegStream.frame_to_timestamp(self.video_info.length, self.video_info.fps)
            end_timestamp = ''.join(end_timestamp[:-4])
        else:
            end_timestamp = FFmpegStream.frame_to_timestamp(self.params.end_frame, self.video_info.fps)
            end_timestamp = ''.join(end_timestamp[:-4])

        txt = current_timestamp + ' / ' + end_timestamp
        cv2.putText(annotated_img, txt, (max(( 0, img.shape[1] - self.x_text_start - round(len(txt)*17*self.font_size) )), 50),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0,0,255), 2)
        return annotated_img


    def draw_text(self, img: np.ndarray, txt: str, y :int = 50, color :tuple = (0,0,255)) -> np.ndarray:
        """ Draw text to an image/frame

        Args:
            img (np.ndarray): opencv image
            txt (str): the text to plot on the image
            y (int): y position
            colot (tuple): BGR Color tuple

        Returns:
            np.ndarray: opencv image with text
        """
        annotated_img = img.copy()
        cv2.putText(annotated_img, str(txt), (self.x_text_start, y), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, color, 2)
        return annotated_img


    def get_average_tracking_fps(self) -> float:
        """ Calculate current processing FPS

        Returns
            float: FPS
        """
        if len(self.tracking_fps) < 1: return 1
        return sum(self.tracking_fps) / float(len(self.tracking_fps))


    def interpolate_bboxes(self, bboxes :dict) -> None:
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


    def min_max_selector(self,
            image_min :np.ndarray,
            image_max :np.ndarray,
            info :str = "",
            title_min :str = "",
            title_max : str = "",
            lower_limit :int = 0,
            upper_limit :int = 99) -> tuple:
        """ Min Max selection Window

        Args:
            image_min (np.ndarray): the frame/image with lowest position
            image_max (np.ndarray): the frame/image with highest position
            info (str): additional info string th show on the Window
            title_min (str): title for the min selection
            title_max (str): title for the max selection
            lower_limit (int): the lower possible value
            upper_limit (int): the highest possible value

        Returns:
            tuple: with selected (min: flaot, max float)
        """
        cv2.createTrackbar("Min", self.window_name, lower_limit, upper_limit, lambda _: None)
        cv2.createTrackbar("Max", self.window_name, upper_limit, upper_limit, lambda _: None)
        image = np.concatenate((image_min, image_max), axis=1)

        if info != "":
            cv2.putText(image, "Info: "+info, (self.x_text_start, 75), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (255,0,0), 2)

        if title_min != "":
            cv2.putText(image, title_min, (self.x_text_start, 25), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (255,0,0), 2)

        if title_max != "":
            cv2.putText(image, title_max, (image_min.shape[1] + self.x_text_start, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (255,0,0), 2)

        cv2.putText(image, "Use 'space' to quit and set the trackbar values",
            (self.x_text_start, 100), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (255,0,0), 2)

        self.beep()
        self.clear_keypress_queue()
        trackbarValueMin = lower_limit
        trackbarValueMax = upper_limit
        self.logger.info("Waiting for user input")
        while True:
            try:
                preview = image.copy()
                cv2.putText(preview, "Set {} to {}".format('Min', trackbarValueMin),
                    (self.x_text_start, 50), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0,0,255), 2)
                cv2.putText(preview, "Set {} to {}".format('Max', trackbarValueMax),
                    (image_min.shape[1] + self.x_text_start, 50), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0,0,255), 2)
                cv2.imshow(self.window_name, self.preview_scaling(preview, 1.1))
                if self.was_space_pressed() or cv2.waitKey(25) == ord(' '): break
                trackbarValueMin = cv2.getTrackbarPos("Min", self.window_name)
                trackbarValueMax = cv2.getTrackbarPos("Max", self.window_name)
            except: pass

        if platform.system() != 'Windows':
            # TODO logging here on windows cause open background process
            self.logger.info("Receive User Input")
        self.__show_loading_screen(preview.shape)
        return (trackbarValueMin, trackbarValueMax) if trackbarValueMin < trackbarValueMax else (trackbarValueMax, trackbarValueMin)


    def beep(self) -> None:
        """ Play an sound to signal an event """
        if NOTIFICATION_SOUND_FILE is not None:
            if os.path.exists(NOTIFICATION_SOUND_FILE):
                def play_beep():
                    try: playsound(NOTIFICATION_SOUND_FILE)
                    except: pass
                try:
                    self.beep_thread = threading.Thread(target=play_beep)
                    self.beep_thread.start()
                except: pass
            else:
                self.logger.warning("Notification sound file not found (%s)", NOTIFICATION_SOUND_FILE)


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
                'roll': [np.array([]) for _ in range(self.params.number_of_trackers)]
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
                        score['roll'][tracker_number] = np.append(score['roll'][tracker_number], np.arctan(np.array(y / max((10e-3, x)))))
                    elif x >= 0 and y < 0:
                        score['roll'][tracker_number] = np.append(score['roll'][tracker_number], -1.0*np.arctan(np.array(y / max((10e-3, x)))))
                    elif x < 0 and y < 0:
                        score['roll'][tracker_number] = np.append(score['roll'][tracker_number], math.pi + -1.0*np.arctan(np.array(y / x)))
                    elif x < 0 and y >= 0:
                        score['roll'][tracker_number] = np.append(score['roll'][tracker_number], math.pi + np.arctan(np.array(y / x)))
                    else:
                        # this should never happen
                        self.logger.error('Calculate score not implement for x=%d, y=%d', x, y)

                # invert because math angle is ccw
                score['roll'][tracker_number] = np.array([-1.0*item for item in self.score['roll']])


            else:
                score['x'][tracker_number] = np.array([w[0] - min([x[0] for x in bboxes['Woman'][tracker_number]]) for w in bboxes['Woman'][tracker_number]])
                score['y'][tracker_number] = np.array([max([x[1] for x in bboxes['Woman'][tracker_number]]) - w[1] for w in bboxes['Woman'][tracker_number]])

        self.logger.info("Merge Scores")

        if True:
            pool, queue = {}, {}
            for metric in score.keys():
                queue[metric] = mp.Queue()
                pool[metric] = threading.Thread(target=merge_score, args=(score[metric], self.params.number_of_trackers, queue[metric], ))
                pool[metric].start()

            for metric in score.keys():
                pool[metric].join()
                score[metric] = queue[metric].get()
        else:
            for metric in score.keys():
                score[metric] = merge_score(score[metric], self.params.number_of_trackers)

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

            (desired_min, desired_max) = self.min_max_selector(
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


    def plot_y_score(self, name: str, idx_list: list, dpi : int = 300) -> None:
        """ Plot the score to an figure

        Args:
            name (str): file name for the figure
            idx_list (list): list with all frame numbers with funscript action points
            dpi (int): picture output dpi
        """
        if len(self.score['y']) < 2: return
        if len(idx_list) < 2: return
        rows = 2
        figure = Figure(figsize=(max([6,int(len(self.score['y'])/50)]), rows*3+1), dpi=dpi)
        ax = figure.add_subplot(2,1,1) # Rows, Columns, Position
        ax.title.set_text('Motion in y direction')
        # TODO why is there an offset of 1 in the data?
        ax.plot(self.score['y'][max((0,idx_list[0]-1)):idx_list[-1]])
        ax.plot(idx_list, [self.score['y'][idx] for idx in idx_list], 'o')
        ax.legend(['Tracker Prediction','Local Max and Min'], loc='upper right')
        ax = figure.add_subplot(2,1,2)
        ax.title.set_text('Funscript')
        ax.plot(idx_list, [self.score['y'][idx] for idx in idx_list])
        ax.plot(idx_list, [self.score['y'][idx] for idx in idx_list], 'o')
        figure.savefig(fname=name, dpi=dpi, bbox_inches='tight')


    def plot_scores(self, name: str, dpi : int = 300) -> None:
        """ Plot the score to an figure

        Args:
            name (str): file name for the figure
            dpi (int): picture output dpi
        """
        if len(self.score['y']) < 2: return
        rows = 2
        figure = Figure(figsize=(max([6,int(len(self.score['y'])/50)]), rows*3+1), dpi=dpi)
        ax = figure.add_subplot(2,1,1) # Rows, Columns, Position
        ax.title.set_text('Motion in x direction')
        ax.plot(self.score['x'])
        ax = figure.add_subplot(2,1,2)
        ax.title.set_text('Motion in y direction')
        ax.plot(self.score['y'])
        figure.savefig(fname=name, dpi=dpi, bbox_inches='tight')


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


    def preview_scaling(self, preview_image :np.ndarray, post_scale :float = 1.0) -> np.ndarray:
        """ Scale image for preview

        Args:
            preview_image (np.ndarray): opencv image
            post_scale (float, optional): additional scaling factor

        Returns:
            np.ndarray: scaled opencv image
        """
        return cv2.resize(
                preview_image,
                None,
                fx=self.params.preview_scaling*post_scale,
                fy=self.params.preview_scaling*post_scale
            )


    def get_vr_projection_config(self, image :np.ndarray) -> dict:
        """ Get the projection ROI config form user input

        Args:
            image (np.ndarray): opencv vr 180 or 360 image

        Returns:
            dict: projection config
        """
        config = copy.deepcopy(PROJECTION[self.params.projection])

        self.determine_preview_scaling(config['parameter']['width'], config['parameter']['height'])

        # NOTE: improve processing speed to make this menu more responsive
        if image.shape[0] > 6000 or image.shape[1] > 6000:
            image = cv2.resize(image, None, fx=0.25, fy=0.25)

        if image.shape[0] > 3000 or image.shape[1] > 3000:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

        ui_texte = []
        if "keys" in config.keys():
            for param in config['keys'].keys():
                if param in config['parameter'].keys() and all(x in config["keys"][param].keys() for x in ["increase", "decrease"]):
                    ui_texte.append("Use '{}', '{}' to increase/decrease {}".format(
                        config["keys"][param]["increase"],
                        config["keys"][param]["decrease"],
                        param)
                    )

        self.clear_keypress_queue()
        parameter_changed, selected = True, False
        while not selected:
            if parameter_changed:
                parameter_changed = False
                preview = FFmpegStream.get_projection(image, config)

                preview = self.draw_text(preview, "Press 'space' to use current selected region of interest)",
                        y = 50, color = (255, 0, 0))
                preview = self.draw_text(preview, "Press '0' (NULL) to reset view)",
                        y = 75, color = (255, 0, 0))
                for line, txt in enumerate(ui_texte):
                    preview = self.draw_text(preview, txt, y = 100 + (line * 25), color = (0, 255, 0))

            cv2.imshow(self.window_name, self.preview_scaling(preview))

            while self.keypress_queue.qsize() > 0:
                pressed_key = '{0}'.format(self.keypress_queue.get())
                if pressed_key == "Key.space":
                    selected = True
                    break

                if pressed_key == "'0'":
                    config = copy.deepcopy(PROJECTION[self.params.projection])
                    parameter_changed = True
                    break

                if "keys" not in config.keys():
                    break

                for param in config['keys'].keys():
                    if param in config['parameter'].keys() and all(x in config["keys"][param].keys() for x in ["increase", "decrease"]):
                        if pressed_key == "'" + config["keys"][param]["increase"] + "'":
                            config['parameter'][param] += 5
                            parameter_changed = True
                            break
                        elif pressed_key == "'" + config["keys"][param]["decrease"] + "'":
                            config['parameter'][param] -= 5
                            parameter_changed = True
                            break

            if cv2.waitKey(1) in [ord(' ')]: break

        self.__show_loading_screen(preview.shape)
        return config


    def __show_loading_screen(self, shape: tuple, txt: str = "Please wait...") -> None:
        """ Show an loading screen

        Args:
            shape (tuple): image shape of loading screen
            txt (str): text to display
        """
        try:
            background = np.full(shape, 0, dtype=np.uint8)
            loading_screen = self.draw_text(background, txt)
            cv2.imshow(self.window_name, self.preview_scaling(loading_screen))
            cv2.waitKey(1)
        except: pass


    def get_bbox(self, image: np.ndarray, txt: str) -> tuple:
        """ Window to get an initial tracking box (ROI)

        Args:
            image (np.ndarray): opencv image e.g. the first frame to determine the inital tracking box
            txt (str): additional text to display on the selection window

        Returns:
            tuple: the entered box tuple (x,y,w,h)
        """
        image = self.draw_text(image, "Select area with Mouse and Press 'space' or 'enter' to continue",
                y = 75, color = (255, 0, 0))

        if self.params.use_zoom:
            while True:
                zoom_bbox = cv2.selectROI(self.window_name, self.draw_text(image, "Zoom selected area"), False)
                if zoom_bbox is None or len(zoom_bbox) == 0: continue
                if zoom_bbox[2] < 75 or zoom_bbox[3] < 75:
                    self.logger.error("The selected zoom area is to small")
                    continue
                break

            image = image[zoom_bbox[1]:zoom_bbox[1]+zoom_bbox[3], zoom_bbox[0]:zoom_bbox[0]+zoom_bbox[2]]
            image = cv2.resize(image, None, fx=self.params.zoom_factor, fy=self.params.zoom_factor)

        image = self.draw_text(image, txt)
        image = self.preview_scaling(image)
        while True:
            bbox = cv2.selectROI(self.window_name, image, False)
            if bbox is None or len(bbox) == 0: continue
            if bbox[0] == 0 or bbox[1] == 0 or bbox[2] < 9 or bbox[3] < 9: continue
            break

        # revert the preview scaling
        bbox = (round(bbox[0]/self.params.preview_scaling),
                    round(bbox[1]/self.params.preview_scaling),
                    round(bbox[2]/self.params.preview_scaling),
                    round(bbox[3]/self.params.preview_scaling)
                )

        # revert the zoom
        if self.params.use_zoom:
            bbox = (round(bbox[0]/self.params.zoom_factor)+zoom_bbox[0],
                    round(bbox[1]/self.params.zoom_factor)+zoom_bbox[1],
                    round(bbox[2]/self.params.zoom_factor),
                    round(bbox[3]/self.params.zoom_factor)
                )

        return bbox


    def get_flat_projection_config(self,
            first_frame :np.ndarray) -> dict:
        """ Get the flat config parameter

        Args:
            first_frame (np.ndarray): opencv image

        Returns:
            dict: config
        """
        h, w = first_frame.shape[:2]
        config = copy.deepcopy(PROJECTION[self.params.projection])

        if PROJECTION[self.params.projection]['parameter']['height'] == -1:
            scaling = config['parameter']['width'] / float(w)
            config['parameter']['height'] = round(h * scaling)
        elif PROJECTION[self.params.projection]['parameter']['width'] == -1:
            scaling = config['parameter']['height'] / float(h)
            config['parameter']['width'] = round(w * scaling)

        self.determine_preview_scaling(config['parameter']['width'], config['parameter']['height'])

        return config


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
            bbox_woman = self.get_bbox(preview_frame, "Select Woman Feature #" + str(tracker_number+1))
            preview_frame = self.draw_box(preview_frame, bbox_woman, color=(255,0,255))
            if self.params.supervised_tracking:
                while True:
                    tracking_areas_woman[tracker_number] = self.get_bbox(preview_frame, "Select the Supervised Tracking Area for the Woman Feature #" + str(tracker_number+1))
                    if StaticVideoTracker.is_bbox_in_tracking_area(bbox_woman, tracking_areas_woman[tracker_number]): break
                    self.logger.error("Invalid supervised tracking area selected")
                preview_frame = self.draw_box(preview_frame, tracking_areas_woman[tracker_number], color=(0,255,0))
                trackers_woman[tracker_number] = StaticVideoTracker(first_frame, bbox_woman, self.video_info.fps, supervised_tracking_area = tracking_areas_woman[tracker_number])
            else:
                trackers_woman[tracker_number] = StaticVideoTracker(first_frame, bbox_woman, self.video_info.fps)

            if tracker_number == 0:
                bboxes['Woman'][1] = { tracker_number: bbox_woman }
            else:
                bboxes['Woman'][1][tracker_number] = bbox_woman

            if self.params.track_men:
                bbox_men = self.get_bbox(preview_frame, "Select Men Feature #" + str(tracker_number+1))
                preview_frame = self.draw_box(preview_frame, bbox_men, color=(255,0,255))
                if self.params.supervised_tracking:
                    while True:
                        tracking_areas_men[tracker_number] = self.get_bbox(preview_frame, "Select the Supervised Tracking Area for the Men Feature #" + str(tracker_number+1))
                        if StaticVideoTracker.is_bbox_in_tracking_area(bbox_men, tracking_areas_men[tracker_number]): break
                        self.logger.error("Invalid supervised tracking area selected")
                    preview_frame = self.draw_box(preview_frame, tracking_areas_men[tracker_number], color=(255,0,255))
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

        if self.is_vr_video():
            projection_config = self.get_vr_projection_config(first_frame)
        else:
            projection_config = self.get_flat_projection_config(first_frame)

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

        tracking_lost_frames = round(self.video_info.fps * self.params.tracking_lost_time / 1000.0)
        if tracking_lost_frames > 0 and self.params.number_of_trackers > 1:
            self.logger.warning("Delayed Tracking Lost is currently not implemented for multiple trackers (The feature will be disabled)")
            tracking_lost_frames = 0

        if self.params.scene_detector.upper() == "CONTENT":
            scene_detector = SceneContentDetector(self.params.start_frame, first_frame, self.params.skip_frames, self.video_info.fps)
        elif self.params.scene_detector.upper() == "THRESHOLD":
            scene_detector = SceneThresholdDetector(self.params.start_frame, self.params.skip_frames, self.video_info.fps)
        else:
            # Fallback is SceneDetectFromFile
            scene_detector = SceneDetectFromFile(self.params.video_path, self.params.start_frame, self.params.skip_frames, self.video_info.fps)

        status = "End of video reached"
        self.clear_keypress_queue()
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
                    for tracker_number in range(self.params.number_of_trackers):
                        if bbox_woman[tracker_number] is not None:
                            if tracker_number == 0:
                                bboxes['Woman'][frame_num-1] = { tracker_number: bbox_woman[tracker_number] }
                            else:
                                bboxes['Woman'][frame_num-1][tracker_number] = bbox_woman[tracker_number]
                            last_frame = self.draw_box(last_frame, bboxes['Woman'][frame_num-1][tracker_number], color=(0,255,0))
                            if self.params.supervised_tracking:
                                last_frame = self.draw_box(last_frame, tracking_areas_woman[tracker_number], color=(0,255,0))

                        if self.params.track_men and bbox_men[tracker_number] is not None:
                            if tracker_number == 0:
                                bboxes['Men'][frame_num-1] = { tracker_number: bbox_men[tracker_number] }
                            else:
                                bboxes['Men'][frame_num-1][tracker_number] = bbox_men[tracker_number]
                            last_frame = self.draw_box(last_frame, bboxes['Men'][frame_num-1][tracker_number], color=(255,0,255))
                            if self.params.supervised_tracking:
                                last_frame = self.draw_box(last_frame, tracking_areas_men[tracker_number], color=(255,0,255))

                    last_frame = self.draw_fps(last_frame)
                    last_frame = self.draw_time(last_frame, frame_num + self.params.start_frame)

                    scene_change_quit_flag = False
                    if scene_detector.is_scene_change(frame_num-1 + self.params.start_frame):
                        self.logger.info("Scene change detected, Pause tracking")
                        self.beep()
                        cv2.putText(last_frame, "Scene change detected, Press 'space' to continue tracking or press 'q' to finalize tracking",
                                (self.x_text_start, 75), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (255,0,0), 2)
                        cv2.imshow(self.window_name, self.preview_scaling(last_frame))
                        while True:
                            key = cv2.waitKey(25)
                            if self.was_space_pressed() or key == ord(' '):
                                break

                            if self.was_key_pressed('q') or key == ord('q'):
                                scene_change_quit_flag = True
                                break

                    cv2.putText(last_frame, "Press 'q' if the tracking point shifts or a video cut occured",
                            (self.x_text_start, 75), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (255,0,0), 2)
                    cv2.imshow(self.window_name, self.preview_scaling(last_frame))

                    if scene_change_quit_flag:
                        status = 'Tracking stopped at scene change'
                        if self.params.scene_detector.upper() == "THRESHOLD":
                            # NOTE: The threshold scene detector has delayed detection
                            delete_last_predictions = int(self.video_info.fps)
                        else:
                            delete_last_predictions = (self.params.skip_frames+1)*2
                        break

                    if self.was_key_pressed('q') or cv2.waitKey(1) == ord('q'):
                        status = 'Tracking stopped by user'
                        delete_last_predictions = max((1, int((self.get_average_tracking_fps()+1)*0.5*HYPERPARAMETER['user_reaction_time_in_milliseconds']/1000.0)))
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

        self.__show_loading_screen(first_frame.shape)
        self.logger.info("Raw tracking data: %d Tracking points for %d seconds of the video", len(bboxes["Woman"]), int(len(bboxes["Woman"])*(self.params.skip_frames + 1)/self.video_info.fps))
        video.stop()
        bboxes = self.correct_bboxes(bboxes, delete_last_predictions)
        self.logger.info(status)
        self.logger.info('Interpolate tracking boxes')
        interpolated_bboxes = self.interpolate_bboxes(bboxes)
        self.calculate_score(interpolated_bboxes)
        return status


    def clear_keypress_queue(self) -> None:
        """ Clear the key press queue """
        while self.keypress_queue.qsize() > 0:
            self.keypress_queue.get()


    def was_key_pressed(self, key: str) -> bool:
        """ Check if key was presssed

        Args:
            key (str): the key to check

        Returns:
            bool: True if 'q' was pressed else False
        """
        if key is None or len(key) == 0: return False
        while self.keypress_queue.qsize() > 0:
            if '{0}'.format(self.keypress_queue.get()) == "'"+key[0]+"'": return True
        return False


    def was_space_pressed(self) -> bool:
        """ Check if 'space' was presssed

        Returns:
            bool: True if 'space' was pressed else False
        """
        while self.keypress_queue.qsize() > 0:
            if '{0}'.format(self.keypress_queue.get()) == "Key.space": return True
        return False


    def on_key_press(self, key: Key) -> None:
        """ Our key press handle to register the key presses

        Args:
            key (pynput.keyboard.Key): the pressed key
        """
        if not self.keypress_queue.full():
            self.keypress_queue.put(key)


    def finished(self, status: str, success :bool) -> None:
        """ Process necessary steps to complete the predicted funscript

        Args:
            status (str): a process status/error message
            success (bool): True if funscript was generated else False
        """
        try: cv2.destroyWindow(self.window_name)
        except: pass
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
        shift_max = self.params.shift_top_points if metric == 'y' and self.is_vr_video() else self.params.shift_max_points
        shift_min = self.params.shift_bottom_points if metric == 'y' and self.is_vr_video() else self.params.shift_min_points

        if position in ['max'] :
            if frame_number >= -1*shift_max \
                    and frame_number + shift_max < len(self.score[metric]): \
                    return self.params.start_frame + frame_number + shift_max

        if position in ['min']:
            if frame_number >= -1*shift_min \
                    and frame_number + shift_min < len(self.score[metric]): \
                    return self.params.start_frame + frame_number + shift_min

        return self.params.start_frame + frame_number


    def get_score_with_offset(self, idx_dict: dict, metric: str) -> list:
        """ Apply the offsets form config file

        Args:
            idx_dict (dict): the idx dictionary with {'min':[], 'max':[]} idx lists
            metric (str): the metric for the score calculation

        Returns:
            list: score with offset
        """
        offset_max = self.params.top_points_offset if metric == 'y' and self.is_vr_video() else self.params.max_points_offset
        offset_min = self.params.bottom_points_offset if metric == 'y' and self.is_vr_video() else self.params.min_points_offset

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

            threshold_min = self.params.bottom_threshold if self.params.metric == 'y' and self.is_vr_video() else self.params.min_threshold
            threshold_max = self.params.top_threshold if self.params.metric == 'y' and self.is_vr_video() else self.params.max_threshold

            for idx in idx_dict['min']:
                self.funscript.add_action(
                        min(output_score) \
                                if output_score[idx] < min(output_score) + threshold_min \
                                else round(output_score[idx]),
                        FFmpegStream.frame_to_millisec(self.apply_shift(idx, self.params.metric, 'min'), self.video_info.fps)
                    )

            for idx in idx_dict['max']:
                self.funscript.add_action(
                        max(output_score) \
                                if output_score[idx] > max(output_score) - threshold_max \
                                else round(output_score[idx]),
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

            with Listener(on_press=self.on_key_press) as _:
                status = self.tracking()

                if False:
                    with open('debug.json', 'w') as f:
                        json.dump(self.score, f)

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

            if False:
                idx_list = [x for k in ['min', 'max'] for x in idx_dict[k]]
                idx_list.sort()
                self.plot_y_score('debug.png', idx_list)

            self.create_funscript(idx_dict)
            if self.params.invert:
                self.funscript.invert_actions()
            self.finished(status, True)
        except Exception as ex:
            self.logger.critical("The program crashed due to a fatal error", exc_info=ex)
            # self.logger.critical("The program crashed due to a fatal error. " \
            #         + "Please open an issue on github with the corresponding log file (" \
            #         + ','.join(get_logfiles_paths()) + ") and application configuration (" \
            #         + SETTINGS_CONFIG_FILE + ", " + HYPERPARAMETER_CONFIG_FILE + ")", exc_info=ex)
            self.finished("The program crashed due to a fatal error", False)
