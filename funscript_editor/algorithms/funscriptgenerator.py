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
from funscript_editor.data.filevideostream import FileVideoStream
from funscript_editor.algorithms.videotracker import StaticVideoTracker
from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.figure import Figure
from funscript_editor.definitions import VIDEO_SCALING_CONFIG_FILE
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS
from datetime import datetime
from funscript_editor.data.equirectangularvideostream import EquirectangularVideoStream

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
    skip_frames: int = max((0, int(HYPERPARAMETER['skip_frames'])))
    max_playback_fps: int = max((0, int(SETTINGS['max_playback_fps'])))
    direction: str = SETTINGS['tracking_direction']
    use_zoom: bool = SETTINGS['use_zoom']
    shift_bottom_points :int = int(HYPERPARAMETER['shift_bottom_points'])
    shift_top_points :int = int(HYPERPARAMETER['shift_top_points'])
    top_points_offset :float = float(HYPERPARAMETER['top_points_offset'])
    bottom_points_offset :float = float(HYPERPARAMETER['bottom_points_offset'])
    use_equirectangular :bool = SETTINGS['use_equirectangular']
    equirectangular_scaling :float = max((0.2, float(SETTINGS['equirectangular_scaling'])))
    zoom_factor :float = max((1.0, float(SETTINGS['zoom_factor'])))
    scaling_method :str = SETTINGS['scaling_method']
    top_threshold :float = float(HYPERPARAMETER['top_threshold'])
    bottom_threshold :float = float(HYPERPARAMETER['bottom_threshold'])
    equirectangular_height :int = 720
    equirectangular_width :int = 1240
    equirectangular_fov :int = 100


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
        self.timer = cv2.getTickCount()

        # XXX destroyWindow(...) sems not to delete the trackbar. Workaround: we give the window each time a unique name
        self.window_name = "Funscript Generator ({})".format(datetime.now().strftime("%H:%M:%S"))

        # scale config for the video
        with open(VIDEO_SCALING_CONFIG_FILE, 'r') as config_file:
            self.scale = json.load(config_file)
            self.scale = {int(k) : float(v) for k,v in self.scale.items()}

        self.keypress_queue = Queue(maxsize=32)
        self.stopped = False
        self.perspective_params = {}
        self.x_text_start = 50
        self.scone_x = []
        self.scone_y = []
        self.bboxes = {
                'Men': [],
                'Woman': []
                }

        cap = cv2.VideoCapture(self.params.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.tracking_fps = []
        cap.release()

    #: completed event with reference to the funscript with the predicted actions, status message and success flag
    funscriptCompleted = QtCore.pyqtSignal(object, str, bool)

    #: processing event with current processed frame number
    processStatus = QtCore.pyqtSignal(int)

    __logger = logging.getLogger(__name__)

    def get_scaling(self, frame_width: int, frame_height: int) -> float:
        """ Get the scaling parameter for current video

        Args:
            frame_width (int): frame width of current video
            frame_height (int): frame height of current video

        Returns:
            float: scaling parameter from scaling config
        """
        if self.params.scaling_method == 'auto':
            scale = []
            try:
                for monitor in get_monitors():
                    scale.append( min((monitor.width / (frame_width*1.1), monitor.height / (frame_height*1.1) )) )
            except: pass

            if len(scale) == 0:
                self.__logger.error("Monitor info not found, please switch to scaling_method: 'config'")
                scale = 1.0
            else:
                # asume we use the largest monitor for scipting
                scale = max(scale)
        else:
            # assume scaling_method is 'config' (fallback)
            scale = max([0.05, min([8.0]+[self.scale[k] for k in self.scale.keys() if k < frame_width])])

        self.__logger.info("Use scaling of %f for preview window and tracking", scale)
        return scale

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
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        self.timer = cv2.getTickCount()
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
        cv2.putText(annotated_img, str(txt), (self.x_text_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return annotated_img


    def get_average_tracking_fps(self) -> float:
        """ Calculate current processing FPS

        Returns
            float: FPS
        """
        if len(self.tracking_fps) < 1: return 1
        return sum(self.tracking_fps) / float(len(self.tracking_fps))


    def append_interpolated_bbox(self, bbox :tuple, target: str) -> None:
        """ Interpolate tracking boxes for skiped frames

        Args:
            bbox (tuple): the new tracking box (x,y,w,h)
            target (str): the target where to save the interpolated tracking boxes
        """
        if self.params.skip_frames > 0 and len(self.bboxes[target]) > 0:
            for i in range(1, self.params.skip_frames+1):
                x0 = np.interp(i, [0, self.params.skip_frames+1], [self.bboxes[target][-1][0], bbox[0]])
                y0 = np.interp(i, [0, self.params.skip_frames+1], [self.bboxes[target][-1][1], bbox[1]])
                w = np.interp(i, [0, self.params.skip_frames+1], [self.bboxes[target][-1][2], bbox[2]])
                h = np.interp(i, [0, self.params.skip_frames+1], [self.bboxes[target][-1][3], bbox[3]])
                self.bboxes[target].append((x0, y0, w, h))
        self.bboxes[target].append(bbox)


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
            cv2.putText(image, "Info: "+info, (self.x_text_start, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        if title_min != "":
            cv2.putText(image, title_min, (self.x_text_start, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        if title_max != "":
            cv2.putText(image, title_max, (image_min.shape[1] + self.x_text_start, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.putText(image, "Use 'space' to quit and set the trackbar values",
            (self.x_text_start, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        self.clear_keypress_queue()
        trackbarValueMin = lower_limit
        trackbarValueMax = upper_limit
        while True:
            try:
                preview = image.copy()
                cv2.putText(preview, "Set {} to {}".format('Min', trackbarValueMin),
                    (self.x_text_start, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(preview, "Set {} to {}".format('Max', trackbarValueMax),
                    (image_min.shape[1] + self.x_text_start, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow(self.window_name, preview)
                if self.was_space_pressed() or cv2.waitKey(25) == ord(' '): break
                trackbarValueMin = cv2.getTrackbarPos("Min", self.window_name)
                trackbarValueMax = cv2.getTrackbarPos("Max", self.window_name)
            except: pass

        return (trackbarValueMin, trackbarValueMax) if trackbarValueMin < trackbarValueMax else (trackbarValueMax, trackbarValueMin)


    def calculate_score(self) -> None:
        """ Calculate the score for the predicted tracking boxes

        Note:
            We use x0,y0 from the predicted tracking boxes to create a diff score
        """
        if self.params.track_men:
            self.score_x = [m[0] - w[0] for w, m in zip(self.bboxes['Woman'], self.bboxes['Men'])]
            self.score_y = [m[1] - w[1] for w, m in zip(self.bboxes['Woman'], self.bboxes['Men'])]
        else:
            self.score_x = [max([x[0] for x in self.bboxes['Woman']]) - w[0] for w in self.bboxes['Woman']]
            self.score_y = [max([x[1] for x in self.bboxes['Woman']]) - w[1] for w in self.bboxes['Woman']]

        # NOTE: Scaling with the anomaly function sometimes makes the result even worse
        # self.score_x = sp.scale_signal_with_anomalies(self.score_x, 0, 100, lower_quantile=0.05, upper_quantile=0.99999)
        # self.score_y = sp.scale_signal_with_anomalies(self.score_y, 0, 100, lower_quantile=0.05, upper_quantile=0.99999)

        self.score_x = sp.scale_signal(self.score_x, 0, 100)
        self.score_y = sp.scale_signal(self.score_y, 0, 100)


    @staticmethod
    def frame_to_millisec(frame: int, fps: float) -> int:
        """ Convert frame number to timestamp in video

        Args:
            frame (int): the framenumber to convert to an timestamp
            fps (float): Video FPS

        Returns:
            int: the timestamp in milliseconds in video for the given framenumber
        """
        if frame < 0: return 0
        return int(round(float(frame)*float(1000)/fps))


    def scale_score(self, status: str, direction : str = 'y') -> None:
        """ Scale the score to desired stroke high

        Note:
            We determine the lowerst and highes positions in the score and request the real position from user.

        Args:
            status (str): a status/info message to display in the window
            direction (str): scale the 'y' or 'x' score
        """
        if len(self.score_y) < 2: return

        cap = cv2.VideoCapture(self.params.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if direction == 'x':
            min_frame = np.argmin(np.array(self.score_x)) + self.params.start_frame
            max_frame = np.argmax(np.array(self.score_x)) + self.params.start_frame
        else:
            min_frame = np.argmin(np.array(self.score_y)) + self.params.start_frame
            max_frame = np.argmax(np.array(self.score_y)) + self.params.start_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
        successMin, imgMin = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_frame)
        successMax, imgMax = cap.read()

        cap.release()

        if successMin and successMax:
            if self.params.use_equirectangular:
                dim =  (int(self.params.equirectangular_width*self.params.equirectangular_scaling), \
                        int(self.params.equirectangular_height*self.params.equirectangular_scaling))
                imgMin = cv2.resize(imgMin, dim)
                imgMax = cv2.resize(imgMax, dim)
            else:
                scale = self.get_scaling(width, height)
                imgMin = cv2.resize(imgMin, None, fx=scale, fy=scale)
                imgMax = cv2.resize(imgMax, None, fx=scale, fy=scale)

            # Assume we have VR 3D Side by Side
            imgMin = imgMin[:, :int(imgMin.shape[1]/2)]
            imgMax = imgMax[:, :int(imgMax.shape[1]/2)]

            (desired_min, desired_max) = self.min_max_selector(
                    image_min = imgMin,
                    image_max = imgMax,
                    info = status,
                    title_min = str("Bottom" if direction != "x" else "Left"),
                    title_max = ("Top" if direction != "x" else "Right")
                )
        else:
            self.__logger.warning("Determine min and max failed")
            desired_min = 0
            desired_max = 99

        if direction == 'x':
            self.score_x = sp.scale_signal(self.score_x, desired_min, desired_max)
        else:
            self.score_y = sp.scale_signal(self.score_y, desired_min, desired_max)


    def plot_y_score(self, name: str, idx_list: list, dpi : int = 300) -> None:
        """ Plot the score to an figure

        Args:
            name (str): file name for the figure
            idx_list (list): list with all frame numbers with funscript action points
            dpi (int): picture output dpi
        """
        if len(self.score_y) < 2: return
        if len(idx_list) < 2: return
        rows = 2
        figure = Figure(figsize=(max([6,int(len(self.score_y)/50)]), rows*3+1), dpi=dpi)
        ax = figure.add_subplot(2,1,1) # Rows, Columns, Position
        ax.title.set_text('Motion in y direction')
        # TODO why is there an offset of 1 in the data?
        ax.plot(self.score_y[max((0,idx_list[0]-1)):idx_list[-1]])
        ax.plot(idx_list, [self.score_y[idx] for idx in idx_list], 'o')
        ax.legend(['Tracker Prediction','Local Max and Min'], loc='upper right')
        ax = figure.add_subplot(2,1,2)
        ax.title.set_text('Funscript')
        ax.plot(idx_list, [self.score_y[idx] for idx in idx_list])
        ax.plot(idx_list, [self.score_y[idx] for idx in idx_list], 'o')
        figure.savefig(fname=name, dpi=dpi, bbox_inches='tight')


    def plot_scores(self, name: str, dpi : int = 300) -> None:
        """ Plot the score to an figure

        Args:
            name (str): file name for the figure
            dpi (int): picture output dpi
        """
        if len(self.score_y) < 2: return
        rows = 2
        figure = Figure(figsize=(max([6,int(len(self.score_y)/50)]), rows*3+1), dpi=dpi)
        ax = figure.add_subplot(2,1,1) # Rows, Columns, Position
        ax.title.set_text('Motion in x direction')
        ax.plot(self.score_x)
        ax = figure.add_subplot(2,1,2)
        ax.title.set_text('Motion in y direction')
        ax.plot(self.score_y)
        figure.savefig(fname=name, dpi=dpi, bbox_inches='tight')


    def delete_last_tracking_predictions(self, num :int) -> None:
        """ Delete the latest tracking predictions e.g. to clear bad tracking values

        Args:
            num (int): number of frames to remove from predicted boxes
        """
        if len(self.bboxes['Woman']) <= num-1:
            self.bboxes['Woman'] = []
            self.bboxes['Men'] = []
        else:
            for i in range(len(self.bboxes['Woman'])-1,len(self.bboxes['Woman'])-num,-1):
                del self.bboxes['Woman'][i]
                if self.params.track_men: del self.bboxes['Men'][i]


    def get_perspective_roi(self, image :np.ndarray) -> None:
        """ Get the perspective ROI parameters form user input

        Args:
            image (np.ndarray): opencv vr 180 or 360 image
        """
        perspective = {
                'FOV': self.params.equirectangular_fov,
                'THETA': -90,
                'PHI': -45,
                'height': int(self.params.equirectangular_height*self.params.equirectangular_scaling),
                'width': int(self.params.equirectangular_width*self.params.equirectangular_scaling)
            }

        # NOTE: improve processing speed to make this menu more responsive
        if image.shape[0] > 6000 or image.shape[1] > 6000:
            image = cv2.resize(image, None, fx=0.25, fy=0.25)

        if image.shape[0] > 3000 or image.shape[1] > 3000:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

        parameter_changed, selected = True, False
        while not selected:
            if parameter_changed:
                parameter_changed = False
                preview = EquirectangularVideoStream.get_perspective(
                        image,
                        perspective['FOV'],
                        perspective['THETA'],
                        perspective['PHI'],
                        perspective['height'],
                        perspective['width']
                    )

                preview = self.drawText(preview, "Press 'q' to use current selected region of interest)",
                        y = 50, color = (255, 0, 0))
                preview = self.drawText(preview, "Use 'w', 's' to move up/down to the region of interest",
                        y = 75, color = (0, 255, 0))

            cv2.imshow(self.window_name, preview)
            while self.keypress_queue.qsize() > 0:
                pressed_key = '{0}'.format(self.keypress_queue.get())
                if pressed_key == "'q'":
                    selected = True
                    break
                elif pressed_key == "'w'":
                    perspective['PHI'] = min((80, perspective['PHI'] + 5))
                    parameter_changed = True
                elif pressed_key == "'s'":
                    perspective['PHI'] = max((-80, perspective['PHI'] - 5))
                    parameter_changed = True
                # elif pressed_key == "'a'":
                #     perspective['THETA'] -= 5
                # elif pressed_key == "'d'":
                #     perspective['THETA'] += 5

            if cv2.waitKey(1) in [ord('q')]: break

        try:
            background = np.full(preview.shape, 255, dtype=np.uint8)
            loading_screen = self.drawText(background, "Please wait ...")
            cv2.imshow(self.window_name, loading_screen)
        except: pass

        self.perspective_params = perspective


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
        # cv2.namedWindow(self.window_name)
        # cv2.createTrackbar("Scale", self.window_name, 1, 10, lambda x: None)

        if self.params.use_zoom:
            while True:
                zoom_bbox = cv2.selectROI(self.window_name, self.drawText(image, "Zoom selected area"), False)
                if zoom_bbox is None or len(zoom_bbox) == 0: continue
                if zoom_bbox[2] < 75 or zoom_bbox[3] < 75:
                    self.__logger.error("The selected zoom area is to small")
                    continue
                break

            image = image[zoom_bbox[1]:zoom_bbox[1]+zoom_bbox[3], zoom_bbox[0]:zoom_bbox[0]+zoom_bbox[2]]
            image = cv2.resize(image, None, fx=self.params.zoom_factor, fy=self.params.zoom_factor)

        while True:
            bbox = cv2.selectROI(self.window_name, self.drawText(image, txt), False)
            if bbox is None or len(bbox) == 0: continue
            if bbox[0] == 0 or bbox[1] == 0 or bbox[2] < 9 or bbox[3] < 9: continue
            break

        if self.params.use_zoom:
            bbox = (round(bbox[0]/self.params.zoom_factor)+zoom_bbox[0],
                    round(bbox[1]/self.params.zoom_factor)+zoom_bbox[1],
                    round(bbox[2]/self.params.zoom_factor),
                    round(bbox[3]/self.params.zoom_factor)
                )

        return bbox


    def get_first_frame(self) -> np.ndarray:
        """ Get the first frame image

        Returns:
            np.ndarray: opencv image
        """
        cap = cv2.VideoCapture(str(self.params.video_path))
        if self.params.start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, self.params.start_frame)
        success, first_frame = cap.read()
        cap.release()
        return first_frame


    def tracking(self) -> str:
        """ Tracking function to track the features in the video

        Returns:
            str: a process status message e.g. 'end of video reached'
        """
        if self.params.use_equirectangular:
            first_frame = self.get_first_frame()
            self.get_perspective_roi(first_frame)

            video = EquirectangularVideoStream(
                    self.params.video_path,
                    self.perspective_params['FOV'],
                    self.perspective_params['THETA'],
                    self.perspective_params['PHI'],
                    self.perspective_params['height'],
                    self.perspective_params['width'],
                    self.params.start_frame
                )
        else:
            video = FileVideoStream(
                video_path=self.params.video_path,
                scale_determiner=self.get_scaling,
                start_frame=self.params.start_frame)

        first_frame = video.read()
        bboxWoman = self.get_bbox(first_frame, "Select Woman Feature")
        trackerWoman = StaticVideoTracker(first_frame, bboxWoman, limit_searchspace = not self.params.use_equirectangular)
        self.bboxes['Woman'].append(bboxWoman)

        if self.params.track_men:
            bboxMen = self.get_bbox(self.drawBox(first_frame, bboxWoman), "Select Men Feature")
            trackerMen = StaticVideoTracker(first_frame, bboxMen, limit_searchspace = not self.params.use_equirectangular)
            self.bboxes['Men'].append(bboxMen)

        if self.params.max_playback_fps > (self.params.skip_frames+1):
            cycle_time_in_ms = (float(1000) / float(self.params.max_playback_fps)) * (self.params.skip_frames+1)
        else:
            cycle_time_in_ms = 0

        status = "End of video reached"
        self.clear_keypress_queue()
        last_frame, frame_num = None, 1 # first frame is was init frame
        while video.isOpen():
            cycle_start = time.time()
            frame = video.read()
            frame_num += 1

            if frame is None:
                status = 'Reach a corrupt video frame'
                break

            # NOTE: Use != 1 to ensure that the first difference is equal to the folowing (reqired for the interpolation)
            if self.params.skip_frames > 0 and frame_num % (self.params.skip_frames + 1) != 1:
                continue

            trackerWoman.update(frame)
            if self.params.track_men: trackerMen.update(frame)
            self.processStatus.emit(frame_num)

            if last_frame is not None:
                # Process data from last step while the next tracking points get predicted.
                # This should improve the whole processing speed, because the tracker run in a seperate thread
                self.append_interpolated_bbox(bboxWoman, 'Woman')
                last_frame = self.drawBox(last_frame, self.bboxes['Woman'][-1])

                if self.params.track_men:
                    self.append_interpolated_bbox(bboxMen, 'Men')
                    last_frame = self.drawBox(last_frame, self.bboxes['Men'][-1])

                last_frame = self.drawFPS(last_frame)
                cv2.putText(last_frame, "Press 'q' if the tracking point shifts or a video cut occured",
                        (self.x_text_start, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.imshow(self.window_name, last_frame)

                if self.was_key_pressed('q') or cv2.waitKey(1) == ord('q'):
                    status = 'Tracking stopped by user'
                    self.delete_last_tracking_predictions(int(self.get_average_tracking_fps()+1)*3)
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


        video.stop()
        self.__logger.info(status)
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
        cv2.destroyWindow(self.window_name)
        self.funscriptCompleted.emit(self.funscript, status, success)
        self.stopped = True


    def apply_shift(self, frame_number, position: str) -> int:
        """ Apply shift to predicted frame positions

        Args:
            position (str): is max or min
        """
        if position in ['max', 'top'] and self.params.direction != 'x':
            if frame_number >= -1*self.params.shift_top_points \
                    and frame_number + self.params.shift_top_points < len(self.score_y): \
                    return self.params.start_frame + frame_number + self.params.shift_top_points

        if position in ['min', 'bottom'] and self.params.direction != 'x':
            if frame_number >= -1*self.params.shift_bottom_points \
                    and frame_number + self.params.shift_bottom_points < len(self.score_y): \
                    return self.params.start_frame + frame_number + self.params.shift_bottom_points

        return self.params.start_frame + frame_number

    def get_score_with_offset(self, idx_dict) -> list:
        """ Apply the offsets form config file

        Args:
            idx_dict (dict): the idx dictionary with {'min':[], 'max':[]} idx lists

        Returns:
            list: score with offset
        """
        if self.params.direction == 'x':
            return self.score_x

        score = copy.deepcopy(self.score_y)
        score_min, score_max = min(score), max(score)
        for idx in idx_dict['min']:
            score[idx] = max(( score_min, min((score_max, score[idx] + self.params.bottom_points_offset)) ))

        for idx in idx_dict['max']:
            score[idx] = max(( score_min, min((score_max, score[idx] + self.params.top_points_offset)) ))

        return score


    def run(self) -> None:
        """ The Funscript Generator Thread Function """
        # NOTE: score_y and score_x should have the same number size so it should be enouth to check one score length
        with Listener(on_press=self.on_key_press) as listener:
            status = self.tracking()
            if len(self.score_y) >= HYPERPARAMETER['min_frames']:
                if self.params.direction != 'x':
                    self.scale_score(status, direction='y')
                else:
                    self.scale_score(status, direction='x')

        if len(self.score_y) < HYPERPARAMETER['min_frames']:
            self.finished(status + ' -> Tracking time insufficient', False)
            return

        if self.params.direction != 'x':
            idx_dict = sp.get_local_max_and_min_idx(self.score_y, self.fps)
        else:
            idx_dict = sp.get_local_max_and_min_idx(self.score_x, self.fps)

        idx_list = [x for k in ['min', 'max'] for x in idx_dict[k]]
        idx_list.sort()

        if False:
            self.plot_scores('debug_001.png')
            if self.params.direction != 'x':
                self.plot_y_score('debug_002.png', idx_list)

        output_score = self.get_score_with_offset(idx_dict)
        for idx in idx_dict['min']:
            self.funscript.add_action(
                    min(output_score) \
                            if output_score[idx] < min(output_score) + self.params.bottom_threshold \
                            else round(output_score[idx]),
                    self.frame_to_millisec(self.apply_shift(idx, 'min'), self.fps)
                )

        for idx in idx_dict['max']:
            self.funscript.add_action(
                    max(output_score) \
                            if output_score[idx] > max(output_score) - self.params.top_threshold \
                            else round(output_score[idx]),
                    self.frame_to_millisec(self.apply_shift(idx, 'max'), self.fps)
                )

        self.finished(status, True)
