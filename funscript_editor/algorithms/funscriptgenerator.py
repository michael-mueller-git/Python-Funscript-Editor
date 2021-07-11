""" Top level process to generate the funscript actions by tracking selected features in the video """

import cv2
import json
import copy
import time
import logging

from screeninfo import get_monitors
from threading import Thread
from queue import Queue
from pynput.keyboard import Key, Listener
from dataclasses import dataclass
from funscript_editor.data.funscript import Funscript
from funscript_editor.algorithms.videotracker import StaticVideoTracker
from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.figure import Figure
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS, PROJECTION
from datetime import datetime
from funscript_editor.data.ffmpegstream import FFmpegStream, VideoInfo
from funscript_editor.algorithms.kalmanfilter import KalmanFilter2D
from scipy.interpolate import interp1d

import funscript_editor.algorithms.signalprocessing as sp
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class FunscriptGeneratorParameter:
    """ Funscript Generator Parameter Dataclass with default values """
    video_path: str # no default value
    start_frame: int = 0 # default is video start (input: set current video position)
    end_frame: int = -1 # default is video end (-1)
    track_men: bool = True # set by userinput at start (message box)
    raw_output: bool = False

    # Settings
    max_playback_fps: int = max((0, int(SETTINGS['max_playback_fps'])))
    direction: str = SETTINGS['tracking_direction']
    use_zoom: bool = SETTINGS['use_zoom']
    zoom_factor: float = max((1.0, float(SETTINGS['zoom_factor'])))
    preview_scaling: float = float(SETTINGS['preview_scaling'])
    projection: str = str(SETTINGS['projection']).lower()
    use_kalman_filter: bool = SETTINGS['use_kalman_filter']

    # General
    skip_frames: int = max((0, int(HYPERPARAMETER['skip_frames'])))

    # y-Movement
    shift_bottom_points: int = int(HYPERPARAMETER['shift_bottom_points'])
    shift_top_points: int = int(HYPERPARAMETER['shift_top_points'])
    bottom_points_offset: float = float(HYPERPARAMETER['bottom_points_offset'])
    top_points_offset: float = float(HYPERPARAMETER['top_points_offset'])
    bottom_threshold: float = float(HYPERPARAMETER['bottom_threshold'])
    top_threshold: float = float(HYPERPARAMETER['top_threshold'])

    # x-Movement
    shift_left_points: int = int(HYPERPARAMETER['shift_left_points'])
    shift_right_points: int = int(HYPERPARAMETER['shift_right_points'])
    left_points_offset: float = float(HYPERPARAMETER['left_points_offset'])
    right_points_offset: float = float(HYPERPARAMETER['right_points_offset'])
    left_threshold: float = float(HYPERPARAMETER['left_threshold'])
    right_threshold: float = float(HYPERPARAMETER['right_threshold'])


class FunscriptGenerator(QtCore.QThread):
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
                'd': []
            }
        self.bboxes = {
                'Men': [],
                'Woman': []
            }


    #: completed event with reference to the funscript with the predicted actions, status message and success flag
    funscriptCompleted = QtCore.pyqtSignal(object, str, bool)

    #: processing event with current processed frame number
    processStatus = QtCore.pyqtSignal(int)

    logger = logging.getLogger(__name__)


    def determine_preview_scaling(self, frame_width, frame_height) -> float:
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


    def drawBox(self, img: np.ndarray, bbox: tuple) -> np.ndarray:
        """ Draw an tracking box on the image/frame

        Args:
            img (np.ndarray): opencv image
            bbox (tuple): tracking box with (x,y,w,h)

        Returns:
            np.ndarray: opencv image with annotated tracking box
        """
        annotated_img = img.copy()
        cv2.rectangle(annotated_img, (bbox[0], bbox[1]), ((bbox[0]+bbox[2]), (bbox[1]+bbox[3])), (255, 0, 255), 3, 1)
        return annotated_img


    def drawFPS(self, img: np.ndarray) -> np.ndarray:
        """ Draw processing FPS on the image/frame

        Args:
            img (np.ndarray): opencv image

        Returns:
            np.ndarray: opencv image with FPS Text
        """
        annotated_img = img.copy()
        fps = (self.params.skip_frames+1)*cv2.getTickFrequency()/(cv2.getTickCount()-self.timer)
        self.tracking_fps.append(fps)
        cv2.putText(annotated_img, str(int(fps)) + ' fps', (self.x_text_start, 50),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (0,0,255), 2)
        self.timer = cv2.getTickCount()
        return annotated_img


    def drawTime(self, img: np.ndarray, frame_num: int) -> np.ndarray:
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


    def drawText(self, img: np.ndarray, txt: str, y :int = 50, color :tuple = (0,0,255)) -> np.ndarray:
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
            bboxes (dict): the new tracking box (x,y,w,h) in dict {Men: {frame_num: box, ...}, Woman: {frame_num: box, ...}}
        """
        for key in bboxes:
            x = [k for k in bboxes[key].keys()]
            boxes = [v for v in bboxes[key].values()]
            if len(boxes) == 0: continue

            fx0 = interp1d(x, [item[0] for item in boxes], kind = 'quadratic')
            fy0 = interp1d(x, [item[1] for item in boxes], kind = 'quadratic')
            fw  = interp1d(x, [item[2] for item in boxes], kind = 'quadratic')
            fh  = interp1d(x, [item[3] for item in boxes], kind = 'quadratic')

            for i in range(min(x), max(x)+1):
                self.bboxes[key].append((float(fx0(i)), float(fy0(i)), float(fw(i)), float(fh(i))))


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
        cv2.createTrackbar("Min", self.window_name, lower_limit, upper_limit, lambda x: None)
        cv2.createTrackbar("Max", self.window_name, upper_limit, upper_limit, lambda x: None)
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

        self.clear_keypress_queue()
        trackbarValueMin = lower_limit
        trackbarValueMax = upper_limit
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

        self.__show_loading_screen(preview.shape)
        return (trackbarValueMin, trackbarValueMax) if trackbarValueMin < trackbarValueMax else (trackbarValueMax, trackbarValueMin)


    def calculate_score(self) -> None:
        """ Calculate the score for the predicted tracking boxes

        Note:
            We use x0,y0 from the predicted tracking boxes to create a diff score
        """
        if self.params.track_men:
            self.score['x'] = [w[0] - m[0] for w, m in zip(self.bboxes['Woman'], self.bboxes['Men'])]
            self.score['y'] = [m[1] - w[1] for w, m in zip(self.bboxes['Woman'], self.bboxes['Men'])]
            self.score['d'] = [np.sqrt(np.sum((np.array(m[:2]) - np.array(w[:2])) ** 2, axis=0)) \
                    for w, m in zip(self.bboxes['Woman'], self.bboxes['Men'])]
        else:
            self.score['x'] = [w[0] - min([x[0] for x in self.bboxes['Woman']]) for w in self.bboxes['Woman']]
            self.score['y'] = [max([x[1] for x in self.bboxes['Woman']]) - w[1] for w in self.bboxes['Woman']]
            # TODO: how to calc d?

        self.score['x'] = sp.scale_signal(self.score['x'], 0, 100)
        self.score['y'] = sp.scale_signal(self.score['y'], 0, 100)
        self.score['d'] = sp.scale_signal(self.score['d'], 0, 100)


    def scale_score(self, status: str, direction : str = 'y') -> None:
        """ Scale the score to desired stroke high

        Note:
            We determine the lowerst and highes positions in the score and request the real position from user.

        Args:
            status (str): a status/info message to display in the window
            direction (str): scale the 'y' or 'x' score
        """
        if len(self.score['y']) < 2: return

        cap = cv2.VideoCapture(self.params.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if direction == 'd':
            min_frame = np.argmin(np.array(self.score['d'])) + self.params.start_frame
            max_frame = np.argmax(np.array(self.score['d'])) + self.params.start_frame
        elif direction == 'x':
            min_frame = np.argmin(np.array(self.score['x'])) + self.params.start_frame
            max_frame = np.argmax(np.array(self.score['x'])) + self.params.start_frame
        else:
            min_frame = np.argmin(np.array(self.score['y'])) + self.params.start_frame
            max_frame = np.argmax(np.array(self.score['y'])) + self.params.start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
        successMin, imgMin = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_frame)
        successMax, imgMax = cap.read()

        cap.release()

        if successMin and successMax:
            if 'vr' in self.params.projection.split('_'):
                if 'sbs' in self.params.projection.split('_'):
                    imgMin = imgMin[:, :int(imgMin.shape[1]/2)]
                    imgMax = imgMax[:, :int(imgMax.shape[1]/2)]
                elif 'ou' in self.params.projection.split('_'):
                    imgMin = imgMin[:int(imgMin.shape[0]/2), :]
                    imgMax = imgMax[:int(imgMax.shape[0]/2), :]

            if PROJECTION[self.params.projection]['parameter']['width'] > 0:
                scale = PROJECTION[self.params.projection]['parameter']['width'] / float(1.75*imgMax.shape[1])
            else:
                scale = PROJECTION[self.params.projection]['parameter']['height'] / float(1.75*imgMax.shape[0])
            imgMin = cv2.resize(imgMin, None, fx=scale, fy=scale)
            imgMax = cv2.resize(imgMax, None, fx=scale, fy=scale)

            if direction == 'y':
                title_min = "Bottom"
            elif direction == 'x':
                title_min = "Left"
            else:
                title_min = "Minimum"

            if direction == 'y':
                title_max = "Top"
            elif direction == 'x':
                title_max = "Right"
            else:
                title_max = "Maximum"

            (desired_min, desired_max) = self.min_max_selector(
                    image_min = imgMin,
                    image_max = imgMax,
                    info = status,
                    title_min = title_min,
                    title_max = title_max
                )
        else:
            self.logger.warning("Determine min and max failed")
            desired_min = 0
            desired_max = 99

        if direction == 'd':
            self.score['d'] = sp.scale_signal(self.score['d'], desired_min, desired_max)
        elif direction == 'x':
            self.score['x'] = sp.scale_signal(self.score['x'], desired_min, desired_max)
        else:
            self.score['y'] = sp.scale_signal(self.score['y'], desired_min, desired_max)


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


    def delete_last_tracking_predictions(self, num :int) -> None:
        """ Delete the latest tracking predictions e.g. to clear bad tracking values

        Args:
            num (int): number of frames to remove from predicted boxes
        """
        if len(self.bboxes['Woman']) <= num:
            self.bboxes['Woman'] = []
            self.bboxes['Men'] = []
        else:
            for _ in range(num):
                del self.bboxes['Woman'][-1]
                if self.params.track_men: del self.bboxes['Men'][-1]


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

        self.clear_keypress_queue()
        parameter_changed, selected = True, False
        while not selected:
            if parameter_changed:
                parameter_changed = False
                preview = FFmpegStream.get_projection(image, config)

                preview = self.drawText(preview, "Press 'q' to use current selected region of interest)",
                        y = 50, color = (255, 0, 0))
                preview = self.drawText(preview, "Use 'w', 's' to move up/down to the region of interest",
                        y = 75, color = (0, 255, 0))

            cv2.imshow(self.window_name, self.preview_scaling(preview))

            while self.keypress_queue.qsize() > 0:
                pressed_key = '{0}'.format(self.keypress_queue.get())
                if pressed_key == "'q'":
                    selected = True
                    break
                elif pressed_key == "'w'":
                    config['parameter']['phi'] = min((80, config['parameter']['phi'] + 5))
                    parameter_changed = True
                elif pressed_key == "'s'":
                    config['parameter']['phi'] = max((-80, config['parameter']['phi'] - 5))
                    parameter_changed = True

            if cv2.waitKey(1) in [ord('q')]: break

        self.__show_loading_screen(preview.shape)
        return config


    def __show_loading_screen(self, shape :tuple) -> None:
        """ Show an loading screen

        Args:
            shape (tuple): image shape of loading screen
        """
        try:
            background = np.full(shape, 0, dtype=np.uint8)
            loading_screen = self.drawText(background, "Please wait ...")
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
        image = self.drawText(image, "Press 'space' or 'enter' to continue (sometimes not very responsive)",
                y = 75, color = (255, 0, 0))

        if self.params.use_zoom:
            while True:
                zoom_bbox = cv2.selectROI(self.window_name, self.drawText(image, "Zoom selected area"), False)
                if zoom_bbox is None or len(zoom_bbox) == 0: continue
                if zoom_bbox[2] < 75 or zoom_bbox[3] < 75:
                    self.logger.error("The selected zoom area is to small")
                    continue
                break

            image = image[zoom_bbox[1]:zoom_bbox[1]+zoom_bbox[3], zoom_bbox[0]:zoom_bbox[0]+zoom_bbox[2]]
            image = cv2.resize(image, None, fx=self.params.zoom_factor, fy=self.params.zoom_factor)

        image = self.drawText(image, txt)
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


    def tracking(self) -> str:
        """ Tracking function to track the features in the video

        Returns:
            str: a process status message e.g. 'end of video reached'
        """
        first_frame = FFmpegStream.get_frame(self.params.video_path, self.params.start_frame)

        if 'vr' in self.params.projection.split('_'):
            projection_config = self.get_vr_projection_config(first_frame)
        else:
            projection_config = self.get_flat_projection_config(first_frame)

        video = FFmpegStream(
                video_path = self.params.video_path,
                config = projection_config,
                start_frame = self.params.start_frame
            )

        bboxes = {
                'Men': {},
                'Woman': {}
            }

        first_frame = video.read()
        bboxWoman = self.get_bbox(first_frame, "Select Woman Feature")
        trackerWoman = StaticVideoTracker(first_frame, bboxWoman)
        bboxes['Woman'][1] = bboxWoman

        if self.params.track_men:
            bboxMen = self.get_bbox(self.drawBox(first_frame, bboxWoman), "Select Men Feature")
            trackerMen = StaticVideoTracker(first_frame, bboxMen)
            bboxes['Men'][1] = bboxMen

        if self.params.max_playback_fps > (self.params.skip_frames+1):
            cycle_time_in_ms = (float(1000) / float(self.params.max_playback_fps)) * (self.params.skip_frames+1)
        else:
            cycle_time_in_ms = 0

        status = "End of video reached"
        self.clear_keypress_queue()
        last_frame, frame_num = None, 1 # first frame is init frame
        while video.isOpen():
            cycle_start = time.time()
            frame = video.read()
            frame_num += 1

            if frame is None:
                status = 'Reach a corrupt video frame' if video.isOpen() else 'End of video reached'
                break

            # NOTE: Use != 1 to ensure that the first difference is equal to the folowing (reqired for the interpolation)
            if self.params.skip_frames > 0 and frame_num % (self.params.skip_frames + 1) != 1:
                continue

            if self.params.end_frame > 0 and frame_num + self.params.start_frame >= self.params.end_frame:
                status = "Tracking stop at existing action point"
                break

            trackerWoman.update(frame)
            if self.params.track_men: trackerMen.update(frame)
            self.processStatus.emit(frame_num)

            if last_frame is not None:
                # Process data from last step while the next tracking points get predicted.
                # This should improve the whole processing speed, because the tracker run in a seperate thread
                bboxes['Woman'][frame_num-1] = bboxWoman
                last_frame = self.drawBox(last_frame, bboxes['Woman'][frame_num-1])

                if self.params.track_men:
                    bboxes['Men'][frame_num-1] = bboxMen
                    last_frame = self.drawBox(last_frame, bboxes['Men'][frame_num-1])

                last_frame = self.drawFPS(last_frame)
                cv2.putText(last_frame, "Press 'q' if the tracking point shifts or a video cut occured",
                        (self.x_text_start, 75), cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (255,0,0), 2)
                last_frame = self.drawTime(last_frame, frame_num + self.params.start_frame)
                cv2.imshow(self.window_name, self.preview_scaling(last_frame))

                if self.was_key_pressed('q') or cv2.waitKey(1) == ord('q'):
                    status = 'Tracking stopped by user'
                    self.delete_last_tracking_predictions(int((self.get_average_tracking_fps()+1)*2.0))
                    break

            (successWoman, bboxWoman) = trackerWoman.result()
            if not successWoman:
                status = 'Tracker Woman Lost'
                self.delete_last_tracking_predictions((self.params.skip_frames+1)*3)
                break

            if self.params.track_men:
                (successMen, bboxMen) = trackerMen.result()
                if not successMen:
                    status = 'Tracking Men Lost'
                    self.delete_last_tracking_predictions((self.params.skip_frames+1)*3)
                    break

            last_frame = frame

            if cycle_time_in_ms > 0:
                wait = cycle_time_in_ms - (time.time() - cycle_start)*float(1000)
                if wait > 0: time.sleep(wait/float(1000))

        self.__show_loading_screen(first_frame.shape)
        video.stop()
        self.logger.info(status)
        self.logger.info('Interpolate tracking boxes')
        self.interpolate_bboxes(bboxes)
        self.logger.info('Calculate score')
        self.calculate_score()
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


    def apply_shift(self, frame_number, position: str) -> int:
        """ Apply shift to predicted frame positions

        Args:
            position (str): is max or min
        """
        if self.params.direction == 'd':
            shift_a = self.params.shift_top_points
        elif self.params.direction == 'x':
            shift_a = self.params.shift_right_points
        else:
            shift_a = self.params.shift_top_points

        if self.params.direction == 'd':
            shift_b = self.params.shift_bottom_points
        elif self.params.direction == 'x':
            shift_b = self.params.shift_left_points
        else:
            shift_b = self.params.shift_bottom_points

        if position in ['max', 'top', 'right'] :
            if frame_number >= -1*shift_a \
                    and frame_number + shift_a < len(self.score['y']): \
                    return self.params.start_frame + frame_number + shift_a

        if position in ['min', 'bottom', 'left']:
            if frame_number >= -1*shift_b \
                    and frame_number + shift_b < len(self.score['y']): \
                    return self.params.start_frame + frame_number + shift_b

        return self.params.start_frame + frame_number


    def get_score_with_offset(self, idx_dict) -> list:
        """ Apply the offsets form config file

        Args:
            idx_dict (dict): the idx dictionary with {'min':[], 'max':[]} idx lists

        Returns:
            list: score with offset
        """
        if self.params.direction == 'd':
            offset_a = self.params.top_points_offset
        elif self.params.direction == 'x':
            offset_a = self.params.right_points_offset
        else:
            offset_a = self.params.top_points_offset

        if self.params.direction == 'd':
            offset_b = self.params.bottom_points_offset
        elif self.params.direction == 'x':
            offset_b = self.params.left_points_offset
        else:
            offset_b = self.params.bottom_points_offset

        if self.params.direction == 'd':
            score = copy.deepcopy(self.score['d'])
        elif self.params.direction == 'x':
            score = copy.deepcopy(self.score['x'])
        else:
            score = copy.deepcopy(self.score['y'])

        score_min, score_max = min(score), max(score)

        for idx in idx_dict['max']:
            score[idx] = max(( score_min, min((score_max, score[idx] + offset_a)) ))

        for idx in idx_dict['min']:
            score[idx] = max(( score_min, min((score_max, score[idx] + offset_b)) ))

        return score


    def apply_kalman_filter(self) -> None:
        """ Apply Kalman Filter to the tracking prediction """
        if len(self.bboxes['Woman']) < self.video_info.fps: return

        # TODO: we should use the center of the tracking box not x0, y0 of the box

        self.logger.info("Apply kalman filter")
        kalman = KalmanFilter2D(self.video_info.fps)
        kalman.init(self.bboxes['Woman'][0][0], self.bboxes['Woman'][0][1])
        for idx, item in enumerate(self.bboxes['Woman']):
            prediction = kalman.update(item[0], item[1])
            self.bboxes['Woman'][idx] = (prediction[0], prediction[1], item[2], item[3])

        if self.params.track_men:
            kalman = KalmanFilter2D(self.video_info.fps)
            kalman.init(self.bboxes['Men'][0][0], self.bboxes['Men'][0][1])
            for idx, item in enumerate(self.bboxes['Men']):
                prediction = kalman.update(item[0], item[1])
                self.bboxes['Men'][idx] = (prediction[0], prediction[1], item[2], item[3])


    def determin_change_points(self) -> dict:
        """ Determine all change points

        Returns:
            dict: all local max and min points in score {'min':[idx1, idx2, ...], 'max':[idx1, idx2, ...]}
        """
        self.logger.info("Determine local max and min")
        if self.params.direction == 'd':
            idx_dict = sp.get_local_max_and_min_idx(self.score['d'], self.video_info.fps)
        elif self.params.direction == 'x':
            idx_dict = sp.get_local_max_and_min_idx(self.score['x'], self.video_info.fps)
        else:
            idx_dict = sp.get_local_max_and_min_idx(self.score['y'], self.video_info.fps)
        return idx_dict


    def create_funscript(self, idx_dict: dict) -> None:
        """ Generate the Funscript

        Args:
            idx_dict (dict): dictionary with all local max and min points in score
                             {'min':[idx1, idx2, ...], 'max':[idx1, idx2, ...]}
        """
        if self.params.raw_output:
            if self.params.direction == 'd':
                output_score = copy.deepcopy(self.score['d'])
            elif self.params.direction == 'x':
                output_score = copy.deepcopy(self.score['x'])
            else:
                output_score = copy.deepcopy(self.score['y'])

            for idx in range(len(output_score)):
                self.funscript.add_action(
                        output_score[idx],
                        FFmpegStream.frame_to_millisec(self.apply_shift(idx, 'none'), self.video_info.fps)
                    )

        else:
            output_score = self.get_score_with_offset(idx_dict)

            if self.params.direction == 'd':
                threshold_a = self.params.bottom_threshold
            elif self.params.direction == 'x':
                threshold_a = self.params.left_threshold
            else:
                threshold_a = self.params.bottom_threshold

            if self.params.direction == 'd':
                threshold_b = self.params.top_threshold
            elif self.params.direction == 'x':
                threshold_b = self.params.right_threshold
            else:
                threshold_b = self.params.top_threshold

            for idx in idx_dict['min']:
                self.funscript.add_action(
                        min(output_score) \
                                if output_score[idx] < min(output_score) + threshold_a \
                                else round(output_score[idx]),
                        FFmpegStream.frame_to_millisec(self.apply_shift(idx, 'min'), self.video_info.fps)
                    )

            for idx in idx_dict['max']:
                self.funscript.add_action(
                        max(output_score) \
                                if output_score[idx] > max(output_score) - threshold_b \
                                else round(output_score[idx]),
                        FFmpegStream.frame_to_millisec(self.apply_shift(idx, 'max'), self.video_info.fps)
                    )


    def run(self) -> None:
        """ The Funscript Generator Thread Function """
        if self.params.direction == 'd' and not self.params.track_men:
                self.finished("Tracking with 'd' and no men tracker is not implemented!", False)
                return

        if self.params.raw_output:
            self.logger.warning("Raw output is enabled!")

        # NOTE: score['y'] and score['x'] should have the same number size so it should be enouth to check one score length
        with Listener(on_press=self.on_key_press) as listener:
            status = self.tracking()

            if self.params.use_kalman_filter:
                self.apply_kalman_filter()

            if len(self.score['y']) >= HYPERPARAMETER['min_frames']:
                self.logger.info("Scale score")
                self.scale_score(status, direction=self.params.direction)

        if len(self.score['y']) < HYPERPARAMETER['min_frames']:
            self.finished(status + ' -> Tracking time insufficient', False)
            return

        idx_dict = self.determin_change_points()

        if False:
            idx_list = [x for k in ['min', 'max'] for x in idx_dict[k]]
            idx_list.sort()
            self.plot_y_score('debug.png', idx_list)

        self.create_funscript(idx_dict)
        self.finished(status, True)
