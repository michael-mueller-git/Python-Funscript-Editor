import csv
import os
import logging
import cv2

import numpy as np


class SceneDetectFromFile:
    """ Read the Scenes.csv file from [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).

    Note:
        Use "scenedetect -i video.mkv detect-content list-scenes" command to generate the required Scenes.csv file.


    Args:
        video_path (str): path to current video file
    """

    def __init__(self,
            video_path: str):
        self.logger = logging.getLogger(__name__)
        self.__set_csv_path(video_path)
        self.__load_scenes()


    def __set_csv_path(self, video_path: str) -> None:
        self.scenes_csv_path = str('.'.join(video_path.split('.')[0:-1]) + '-Scenes.csv')


    def __load_scenes(self) -> None:
        self.scenes = []
        if not os.path.exists(self.scenes_csv_path):
            self.logger.info("Scenes CSV File not found")
        else:
            try:
                with open(self.scenes_csv_path) as csvDataFile:
                    csvReader = csv.reader(csvDataFile)
                    found_header = False
                    for row in csvReader:
                        if found_header:
                            # Column 2 (index 1) is Start Frame number
                            try: self.scenes.append(int(row[1]))
                            except: pass
                        elif len(row) > 1 and row[1] == 'Start Frame':
                            found_header = True

                    self.logger.info("Load %d scenes from %s", len(self.scenes), self.scenes_csv_path)
            except Exception as ex:
                self.logger.warning("Scenes CSV has wrong format (%s)!", str(ex))


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
        if frame_number in self.scenes:
            self.logger.info("Detect scene change")
            return True

        return False



class SceneContentDetector:
    """ Detects fast cuts using changes in colour and intensity between frames.

    Args:
        start_frame_number (int): start frame number
        start_frame_img (np.ndarray): start frame opencv image
        threshold (float): thresold value to detect an scene change
        min_scene_len (int): min frames between detected scene changes
    """

    def __init__(self,
            start_frame_number: int,
            start_frame_img: np.ndarray,
            frame_skip_faktor: int,
            threshold: float = 30.0,
            min_scene_len: int = 60):
        self.logger = logging.getLogger(__name__)
        self.start_frame_number = start_frame_number
        self.frame_skip_faktor = frame_skip_faktor
        self.current_frame_number = start_frame_number
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.last_hsv = cv2.split(cv2.cvtColor(start_frame_img, cv2.COLOR_BGR2HSV))
        self.scenes = [start_frame_number]


    def calculate_frame_score(self, curr_hsv) -> float:
        """ Calculate the frame score

        Args:
            curr_hsv (np.array): HSV frame data

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


    def update(self, frame_img: np.ndarray,) -> None:
        """ Update detector

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


    def is_scene_change(self, frame_number: int) -> bool:
        """ Check if current frame number is assosiated with an scene change

        Args:
            frame_number (int): current frame number

        Returns:
            bool: True if current frame belongs to an scene change else False

        """
        if frame_number in self.scenes:
            self.logger.info("Detect scene change")
            return True

        return False
