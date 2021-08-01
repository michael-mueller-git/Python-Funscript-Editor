""" Kalman Filter """

import cv2

import numpy as np


class KalmanFilter2D:
    """ Kalman 2D Filter

    Args:
        fps (float): frames per second
    """

    def __init__(self, fps: float):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.prediction = np.zeros((2, 1), np.float32)
        self.prediction_counter = 0
        self.fps = fps


    def init(self, x: float, y: float) -> None:
        """ Initialize the 2D Kalman Filter

        Args:
            x (float): x measurement
            y (float): y measurement
        """
        # dt = 1.0 / self.fps
        dt = 1.0

        self.kalman.measurementMatrix = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]
                ], np.float32)

        self.kalman.transitionMatrix = np.array(
                [
                    [1, 0, dt,0 ],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0 ],
                    [0, 0, 0, 1 ]
                ], np.float32)

        self.kalman.processNoiseCov = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], np.float32) * 0.03

        self.kalman.correct(np.array([np.float32(x-1), np.float32(y-1)], np.float32))
        self.prediction = self.kalman.predict()
        self.prediction_counter += 1


    def update(self, x: float, y: float) -> list:
        """ Update the 2D Kalman Filter

        Args:
            x (float): x measurement
            y (float): y measurement

        Returns:
            list: prediction [x', y', vx', vy']
        """
        self.kalman.correct(np.array([np.float32(x), np.float32(y)], np.float32))
        self.prediction = self.kalman.predict()
        self.prediction_counter += 1
        # TODO howo to init/fit the OpenCV Kalman Filter?
        if self.prediction_counter < self.fps:
            return [round(x), round(y), 0.0, 0.0]
        else:
            return [round(item[0]) if idx < 2 else float(item[0]) for idx, item in enumerate(self.prediction)]
