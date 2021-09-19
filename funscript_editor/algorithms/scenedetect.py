import csv
import os
import logging
import cv2

import numpy as np

from funscript_editor.utils.config import HYPERPARAMETER


class SceneDetector:
    """ Scene Detector base class

    Args:
        frame_skip_faktor (int): the frame skip faktor of the tracking algorithms
        fps (float): frames per second
        start_frame_number (int): start frame number
    """

    def __init__(self,
            frame_skip_faktor: int,
            fps: int,
            start_frame_number: int):
        self.logger = logging.getLogger(__name__)
        self.frame_skip_faktor = frame_skip_faktor
        self.min_scene_len = round(fps * HYPERPARAMETER['min_scene_len'])
        self.ignore_counter = self.min_scene_len
        self.start_frame_number = start_frame_number
        self.current_frame_number = start_frame_number
        self.scenes = [start_frame_number]


    def update(self, frame_img: np.ndarray) -> None:
        """ Update detector

        Args:
            frame_img (np.ndarray): opencv image frame
        """
        pass


    def is_scene_change(self, frame_number: int) -> bool:
        """ Check if current frame number is assosiated with an scene change

        Args:
            frame_number (int): current frame number

        Returns:
            bool: True if current frame belongs to an scene change else False
        """
        if self.ignore_counter > 0:
            self.ignore_counter -= 1
            return False

        if len(list(filter(lambda x: (x - frame_number) <= 0 and (x - frame_number) > -1*self.frame_skip_faktor, self.scenes))) > 0:
            self.logger.info("Detect scene change")
            self.ignore_counter = self.min_scene_len
            return True

        return False



class SceneDetectFromFile(SceneDetector):
    """ Read the Scenes.csv file from [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).

    Note:
        Use "scenedetect -i videoname.mkv detect-content list-scenes" command to generate the required videoname-Scenes.csv file.


    Args:
        video_path (str): path to current video file
        start_frame_number (int): start frame number
        frame_skip_faktor (int): the frame skip faktor of the tracking algorithms
        fps (float): frames per second
    """

    def __init__(self,
            video_path: str,
            start_frame_number: int,
            frame_skip_faktor: int,
            fps: float):
        super().__init__(frame_skip_faktor, fps, start_frame_number)
        self.logger.info("Use Scene Detector from CSV File")
        self.__load_scenes(video_path)


    def __load_scenes(self, video_path: str) -> None:
        self.scenes_csv_path = str('.'.join(video_path.split('.')[0:-1]) + '-Scenes.csv')
        if not os.path.exists(self.scenes_csv_path):
            self.logger.info("Scenes CSV File not found")
            return

        try:
            with open(self.scenes_csv_path) as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                found_header = False
                for row in csvReader:
                    if found_header:
                        # NOTE: Column 2 (index 1) contains Start Frame number
                        try: self.scenes.append(int(row[1]))
                        except: pass
                    elif len(row) > 1 and row[1] == 'Start Frame':
                        found_header = True

                self.logger.info("Load %d scenes from %s", len(self.scenes), self.scenes_csv_path)
        except Exception as ex:
            self.logger.warning("Scenes CSV has wrong format (%s)!", str(ex))



class SceneContentDetector(SceneDetector):
    """ Detects fast cuts using changes in colour and intensity between frames.

    Args:
        start_frame_number (int): start frame number
        start_frame_img (np.ndarray): start frame opencv image
        frame_skip_faktor (int): the frame skip faktor of the tracking algorithms
        fps (float): frames per second
        threshold (float): thresold value to detect an scene change
    """

    def __init__(self,
            start_frame_number: int,
            start_frame_img: np.ndarray,
            frame_skip_faktor: int,
            fps: float,
            threshold: float = HYPERPARAMETER['scene_content_detector_threshold']):
        super().__init__(frame_skip_faktor, fps, start_frame_number)
        self.logger.info("Use Content Detector")
        self.threshold = threshold
        self.last_hsv = cv2.split(cv2.cvtColor(start_frame_img, cv2.COLOR_BGR2HSV))


    def calculate_frame_score(self, curr_hsv: np.ndarray) -> float:
        """ Calculate the frame score

        Args:
            curr_hsv (np.ndarray): HSV frame data

        Returns:
            float: frame score
        """
        delta_hsv = [0, 0, 0]
        for i in range(3):
            num_pixels = curr_hsv[i].shape[0] * curr_hsv[i].shape[1]
            curr_hsv[i] = curr_hsv[i].astype(np.int32)
            self.last_hsv[i] = self.last_hsv[i].astype(np.int32)
            delta_hsv[i] = np.sum(np.abs(curr_hsv[i] - self.last_hsv[i])) / float(num_pixels)

        self.last_hsv = curr_hsv
        return sum(delta_hsv) / 3.0


    def update(self, frame_img: np.ndarray) -> None:
        """ Update Content Detector

        Args:
            frame_img (np.ndarray): opencv image frame
        """
        self.current_frame_number = self.current_frame_number + self.frame_skip_faktor + 1
        curr_hsv = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))
        frame_score = self.calculate_frame_score(curr_hsv)
        if frame_score >= self.threshold:
            if self.current_frame_number - self.scenes[-1] > self.min_scene_len:
                self.logger.warning("Found scene change at frame %d", self.current_frame_number)
                self.scenes.append(self.current_frame_number)



class SceneThresholdDetector(SceneDetector):
    """ Detects fast cuts/slow fades in from and out to a given threshold level.
    Detects both fast cuts and slow fades so long as an appropriate threshold is chosen.

    Args:
        start_frame_number (int): start frame number
        frame_skip_faktor (int): the frame skip faktor of the tracking algorithms
        fps (float): frames per second
        threshold (int):  8-bit intensity threshold value in order to trigger a fade in/out.
    """

    def __init__(self,
            start_frame_number: int,
            frame_skip_faktor: int,
            fps: float,
            threshold: int = HYPERPARAMETER['scene_threshold_detector_threshold']):
        super().__init__(frame_skip_faktor, fps, start_frame_number)
        self.logger.info("Use Threshold Detector")
        self.threshold = int(threshold)
        self.last_fade = {
            'frame': self.current_frame_number,
            'type': 'in'
        }


    @staticmethod
    def compute_frame_average(frame: np.ndarray) -> float:
        """ Computes the average pixel value/intensity for all pixels in a frame.

        Args:
            frame (np.ndarray): opencv image

        Returns:
            float: average pixel intensity.
        """
        return np.sum(frame[:, :, :]) / float(frame.shape[0] * frame.shape[1] * frame.shape[2])


    def update(self, frame_img: np.ndarray) -> None:
        """ Update Threshold Detector

        Args:
            frame_img (np.ndarray): opencv image frame
        """
        self.current_frame_number = self.current_frame_number + self.frame_skip_faktor + 1
        frame_avg = SceneThresholdDetector.compute_frame_average(frame_img)
        if self.last_fade['type'] == 'in' and frame_avg < self.threshold:
            self.last_fade['type'] = 'out'
            self.last_fade['frame'] = self.current_frame_number
        elif self.last_fade['type'] == 'out' and frame_avg >= self.threshold:
            if (self.current_frame_number - self.scenes[-1]) >= self.min_scene_len:
                self.logger.warning("Found scene change at frame %d", self.current_frame_number)
                self.scenes.append(self.current_frame_number)
            self.last_fade['type'] = 'in'
            self.last_fade['frame'] = self.current_frame_number
