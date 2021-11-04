""" Video Tracker """

from os import supports_effective_ids
import cv2
import time
import logging

from dataclasses import dataclass
from threading import Thread
from queue import Queue
from funscript_editor.utils.config import SETTINGS, HYPERPARAMETER

import numpy as np

class StaticVideoTracker:
    """
    Video tracker for static feature tracking. We assume that the tracked feature is always in
    the neighborhood to the first tracking box. This allows an accelerated prediction.


    Args:
        first_frame (np.ndarray): open cv image representing the start frame
        tracking_bbox (tuple): tuple with (x,y,w,h) of the init tracking box
        fps (float): video fps
        limit_searchspace (dict) : only insert the specified region around the init box
        supervised_tracking_area (tuple, optional): tuple with (x,y,w,h) of the supervised tracking area
        queue_size (int): in (work) and out (result) queue size
    """

    def __init__(self,
            first_frame: np.ndarray,
            tracking_bbox: tuple,
            fps: float,
            limit_searchspace : dict = {'h': 0.45, 'w':0.4},
            supervised_tracking_area: tuple = None,
            queue_size : int = 2):
        self.first_frame = first_frame
        self.limit_searchspace = limit_searchspace
        self.first_tracking_bbox = tracking_bbox
        self.supervised_tracking_area = supervised_tracking_area
        self.fps = max((1,fps))
        self.stopped = False
        self.sleep_time = 0.001
        self.queue_in = Queue(maxsize=queue_size)
        self.queue_out = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()
        self.tracking_points = []
        self.tracking_counter = 0
        self.cluster_center = [0, 0]
        self.plausibility_thresholds = [0, 0]



    __logger = logging.getLogger(__name__)


    @dataclass
    class Status:
        OK :str = "OK"
        TRACKING_LOST :str = "Tracking Lost"
        FEATURE_OUTSIDE :str = "Feature outside the specified area"
        IMPLAUSIBLE :str = "Tracking point is not plausible"


    @staticmethod
    def is_bbox_in_tracking_area(bbox: tuple, supervised_tracking_area: tuple = None) -> bool:
        """ Check if tracking box is inside the supervised tracking area

        Args:
            bbox (tuple): tuple with (x,y,w,h) of the tracking box
            supervised_tracking_area (tuple, optional): tuple with (x,y,w,h) of the supervised tracking area

        Returns:
            bool: True if tracking box is in supervised_tracking_area else False
        """
        if bbox is None:
            return False
        if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
            return False
        if supervised_tracking_area is None:
            return True
        if supervised_tracking_area[2] <= 1:
            return False
        if supervised_tracking_area[3] <= 1:
            return False
        if bbox[0] < supervised_tracking_area[0]:
            return False
        if bbox[1] < supervised_tracking_area[1]:
            return False
        if bbox[0] + bbox[2] > supervised_tracking_area[0] + supervised_tracking_area[2]:
            return False
        if bbox[1] + bbox[3] > supervised_tracking_area[1] + supervised_tracking_area[3]:
            return False

        return True


    def stop(self) -> None:
        """ Stop the tracker thread """
        self.stopped = True
        self.thread.join()


    def update(self, next_frame: np.ndarray) -> None:
        """ Add next frame to work queue

        Args:
            next_frame (np.ndarray): opnecv image representing the next frame
        """
        self.queue_in.put(next_frame)


    def result(self) -> tuple:
        """ Pop the tracking result from result queue

        Returns:
            tuple: predicted tracking box with (x,y,w,h)
        """
        return self.queue_out.get()


    def __setup_tracker(self) -> None:
        """ Setup the tracker specified in the config """
        if SETTINGS['tracker'].upper() == 'MIL':
            self.__logger.info("Start MIL Tracker")
            self.tracker = cv2.TrackerMIL_create()
        elif SETTINGS['tracker'].upper() == 'KCF':
            self.__logger.info("Start KCF Tracker")
            self.tracker = cv2.TrackerKCF_create()
        else:
            # falback is CSRT tracker
            self.__logger.info("Start CSRT Tracker")
            self.tracker = cv2.TrackerCSRT_create()


    def __is_plausible(self, box) -> bool:
        tracking_init_phase_in_sec = HYPERPARAMETER['tracking_init_phase_in_sec']
        tracking_plausibility_factor_x = HYPERPARAMETER['tracking_plausibility_factor_x']
        tracking_plausibility_factor_y = HYPERPARAMETER['tracking_plausibility_factor_y']
        if self.tracking_counter <= round(self.fps * tracking_init_phase_in_sec):
            self.tracking_points.append([box[0] + box[2]/2, box[1] + box[3]/2])
            if self.tracking_counter == round(self.fps * tracking_init_phase_in_sec):
                self.__logger.info("Determine Plausibility Threshold for Tracker")
                self.cluster_center = np.mean(np.array(self.tracking_points), axis = 0)
                distances_x = [abs(self.cluster_center[0] - point[0]) for point in self.tracking_points]
                distances_y = [abs(self.cluster_center[1] - point[1]) for point in self.tracking_points]
                self.plausibility_thresholds = [ \
                        max(distances_x) * tracking_plausibility_factor_x, \
                        max(distances_y) * tracking_plausibility_factor_y \
                ]
        else:
            point = [box[0] + box[2]/2, box[1] + box[3]/2]
            if abs(point[0] - self.cluster_center[0]) > self.plausibility_thresholds[0]:
                self.__logger.warning(
                        "Tracking point x is not plausible (%d > %d +- %d)",
                        round(point[0]),
                        round(self.cluster_center[0]),
                        round(self.plausibility_thresholds[0])
                    )
                return False

            if abs(point[1] - self.cluster_center[1]) > self.plausibility_thresholds[1]:
                self.__logger.warning(
                        "Tracking point y is not plausible (%d > %d +- %d)",
                        round(point[1]),
                        round(self.cluster_center[1]),
                        round(self.plausibility_thresholds[1])
                    )
                return False

            self.cluster_center[0] = (self.cluster_center[0] * self.tracking_counter + point[0]) / (self.tracking_counter + 1)
            self.cluster_center[1] = (self.cluster_center[1] * self.tracking_counter + point[1]) / (self.tracking_counter + 1)

        return True


    def run(self) -> None:
        """ The Video Tracker Thread Function """
        self.__setup_tracker()
        frame_heigt, frame_width = self.first_frame.shape[:2]

        dh, dw = int(frame_heigt*self.limit_searchspace['h']), int(frame_width*self.limit_searchspace['w'])
        x0, y0 = max([0, self.first_tracking_bbox[0] - dw]), max([0, self.first_tracking_bbox[1] - dh])
        y1 = min([frame_heigt, self.first_tracking_bbox[1] + self.first_tracking_bbox[3] + dh])
        x1 = min([frame_width, self.first_tracking_bbox[0] + self.first_tracking_bbox[2] + dw])

        frame_roi = self.first_frame[y0:y1, x0:x1]

        self.tracker.init(
                frame_roi,
                (
                    self.first_tracking_bbox[0] - x0,
                    self.first_tracking_bbox[1] - y0,
                    self.first_tracking_bbox[2],
                    self.first_tracking_bbox[3]
                )
            )

        while not self.stopped:
            if self.queue_in.qsize() == 0 or self.queue_out.full():
                time.sleep(self.sleep_time)
            else:
                frame = self.queue_in.get()
                frame_roi = frame[y0:y1, x0:x1]
                success, bbox = self.tracker.update(frame_roi)
                self.tracking_counter += 1
                status = StaticVideoTracker.Status.TRACKING_LOST
                if not success or bbox is None:
                    bbox = None
                else:
                    status = StaticVideoTracker.Status.OK
                    bbox = (int(bbox[0] + x0), int(bbox[1] + y0), int(bbox[2]), int(bbox[3]))
                    if not StaticVideoTracker.is_bbox_in_tracking_area(bbox, self.supervised_tracking_area):
                        status = StaticVideoTracker.Status.FEATURE_OUTSIDE
                    elif SETTINGS['tracking_plausibility_check']:
                        if not self.__is_plausible(bbox):
                            status = StaticVideoTracker.Status.IMPLAUSIBLE

                self.queue_out.put((status, bbox))

