""" Methods to split equirectangular panorama into normal perspective view. """

import os
import sys
import cv2
import time
import logging

from threading import Thread
from queue import Queue

import numpy as np

from funscript_editor.data.filevideostream import FileVideoStream


class Equirectangular:
    """ Python Class to split equirectangular panorama into normal perspective view.

    NOTE:
        We use the same api as the FileVideoStream to allow MITM

    Args:
        video_stream (FileVideoStream): the file video stream instance
        FOV (int): perspective FOV
        THETA (int): left/right angle in degree (right direction is positive, left direction is negative)
        PHI (int) up/down angle in degree (up direction positive, down direction negative)
        height (int): output image height
        width (int): output image width
        RADIUS (int, optional): sphere radius
    """

    def __init__(self,
            video_stream :FileVideoStream,
            FOV: int,
            THETA: int,
            PHI :int,
            height :int,
            width :int,
            queue_size :int = 256):
        self.video_stream = video_stream
        self.FOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.height = height
        self.width = width
        self.stopped = False
        self.sleep_time = 0.001

        self.Q = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()


    @staticmethod
    def get_perspective(
            img :np.ndarray,
            FOV :int,
            THETA :int,
            PHI :int,
            height :int,
            width :int,
            RADIUS :int = 128) -> np.ndarray:
        """
        Get a normal perspective view from a panorama view.

        Args:
            img (str, or opencv image object): path to image or opencv image data
            FOV (int): perspective FOV
            THETA (int): left/right angle in degree (right direction is positive, left direction is negative)
            PHI (int) up/down angle in degree (up direction positive, down direction negative)
            height (int): output image height
            width (int): output image width
            RADIUS (int, optional): sphere radius

        Returns:
            array: opencv image data
        """
        input_image = cv2.imread(img, cv2.IMREAD_COLOR) if isinstance(img, str) else img
        [input_height, input_width, _] = input_image.shape

        equ_h = input_height
        equ_w = input_width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV if FOV < 180 else 179
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        return cv2.remap(input_image,
                lon.astype(np.float32),
                lat.astype(np.float32),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_WRAP)


    def stop(self) -> None:
        """ Stop equirectangular stream """
        self.video_stream.stop()
        self.stopped = True
        # wait until stream resources are released
        self.thread.join()


    def read(self) -> np.ndarray:
        """ Get next frame from equirectangular stream

        Returns:
            np.ndarray: opencv image data
        """
        return self.Q.get()


    def isOpen(self) -> bool:
        """ Check if equirectangular stream is open or a frame is still available in the buffer

        Returns:
            bool: True if equirectangular strem is open or a frame is still available in the buffer else False
        """
        return self.__more() or not self.stopped


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


    def run(self) -> None:
        """ Function to transform the frames from the file video stream into a queue """
        while not self.stopped and self.video_stream.isOpen():
            if self.Q.full():
                time.sleep(self.sleep_time)
            else:
                frame = self.video_stream.read()
                frame = Equirectangular.get_perspective(
                        frame,
                        self.FOV,
                        self.THETA,
                        self.PHI,
                        self.height,
                        self.width
                    )
                self.Q.put(frame)

        self.stopped = True


    @property
    def current_frame_pos(self) -> int:
        """ Get current frame position

        Returns:
            int: current frame
        """
        return self.video_stream.current_frame_pos


    @property
    def number_of_frames(self) -> int:
        """ Get number of frames in video

        Returns:
            int: number of frames in video
        """
        return self.video_stream.number_of_frames


    @property
    def fps(self) -> float:
        """ Get Video FPS

        Returns:
            float: Video FPS
        """
        return self.video_stream.fps


    @property
    def frame_width(self) -> int:
        """ Get Video Frame Width

        Returns:
            int: video frame width
        """
        return self.video_stream.frame_width


    @property
    def frame_height(self) -> int:
        """ Get Video Frame Height

        Returns:
            int: video frame height
        """
        return self.video_stream.frame_heigt
