""" Video Stream """

import cv2
import time
import logging

from fractions import Fraction
from threading import Thread
from queue import Queue
from typing import Callable

import numpy as np


class FileVideoStream:
    """ Read Video Frames with buffering to improve performance

    Args:
        video_path (str): path to video file
        scale_determiner (Callable[[int, int], float]): function dat determine the scale factor by given frame_width and frame_height
        start_frame (int): start frame number
        queue_size (int): size of frame buffer
    """

    def __init__(self,
            video_path :str,
            scale_determiner :Callable[[int,int], float] = None,
            start_frame :int = 0,
            queue_size :int = 256):

        self.video_path = video_path
        self.start_frame = start_frame
        self.queue_size = queue_size
        self.stopped = False
        self.scale_determiner = scale_determiner
        self.scale = 1.0
        self.sleep_time = 0.001
        self.initialized = False

        self.Q = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()


    __logger = logging.getLogger(__name__)


    def stop(self) -> None:
        """ Stop video stream """
        self.stopped = True
        # wait until stream resources are released
        self.thread.join()


    def read(self) -> np.ndarray:
        """ Get next frame from video stream

        Returns:
            np.ndarray: opencv image data
        """
        return self.Q.get()


    def isOpen(self) -> bool:
        """ Check if file video stream is open or a frame is still available in the buffer

        Returns:
            bool: True if video strem is open or a frame is still available in the buffer else False
        """
        return self.__more() or not self.stopped


    def run(self) -> None:
        """ Function to read frames from the file video stream into a queue """
        self.stream = cv2.VideoCapture(str(self.video_path))
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, self.queue_size)
        if self.start_frame > 0: self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        if self.scale_determiner is not None:
            self.scale = self.scale_determiner(
                    int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )

        self.initialized = True
        while not self.stopped:
            if self.Q.full():
                time.sleep(self.sleep_time)
            else:
                (success, frame) = self.stream.read()

                if not success:
                    self.stopped = True
                    break

                if self.scale and frame is not None and self.scale != 1.0:
                    frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)

                self.Q.put(frame)

        self.stopped = True
        self.stream.release()


    def __more(self) -> bool:
        """ Check if frames in the queue are available

        Returns:
            bool: True if a frame is available else False
        """
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(self.sleep_time)
            tries += 1

        return self.Q.qsize() > 0


    def __wait_for_initialization(self) -> None:
        """ Helper Function to wait for initialization """
        while not self.initialized:
            time.sleep(self.sleep_time)


    @property
    def current_frame_pos(self) -> int:
        """ Get current frame position

        Returns:
            int: current frame
        """
        self.__wait_for_initialization()
        return int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))


    @property
    def number_of_frames(self) -> int:
        """ Get number of frames in video

        Returns:
            int: number of frames in video
        """
        self.__wait_for_initialization()
        return int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))


    @property
    def fps(self) -> float:
        """ Get Video FPS

        Returns:
            float: Video FPS
        """
        self.__wait_for_initialization()
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        if '/' in str(fps): return float(Fraction(fps))
        else: return float(fps)


    @property
    def frame_width(self) -> int:
        """ Get Video Frame Width

        Returns:
            int: video frame width
        """
        self.__wait_for_initialization()
        return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))


    @property
    def frame_height(self) -> int:
        """ Get Video Frame Height

        Returns:
            int: video frame height
        """
        self.__wait_for_initialization()
        return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
