""" Video Tracker """

from os import supports_effective_ids
import cv2
import time
import logging

from threading import Thread
from queue import Queue
from funscript_editor.utils.config import SETTINGS

import numpy as np

class StaticVideoTracker:
    """
    Video tracker for static feature tracking. We assume that the tracked feature is always in
    the neighborhood to the first tracking box. This allows an accelerated prediction.


    Args:
        first_frame (np.ndarray): open cv image representing the start frame
        tracking_bbox (tuple): tuple with (x,y,w,h) of the init tracking box
        limit_searchspace (dict) : only insert the specified region around the init box
        queue_size (int): in (work) and out (result) queue size
    """

    def __init__(self,
            first_frame: np.ndarray,
            tracking_bbox: tuple,
            limit_searchspace : dict = {'h': 0.45, 'w':0.4},
            supervised_tracking_area: tuple = None,
            queue_size : int = 2):
        self.first_frame = first_frame
        self.limit_searchspace = limit_searchspace
        self.first_tracking_bbox = tracking_bbox
        self.supervised_tracking_area = supervised_tracking_area
        self.stopped = False
        self.sleep_time = 0.001
        self.queue_in = Queue(maxsize=queue_size)
        self.queue_out = Queue(maxsize=queue_size)
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()


    __logger = logging.getLogger(__name__)


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
                status = "Tracking Lost"
                if success:
                    status = "OK"
                    bbox = (int(bbox[0] + x0), int(bbox[1] + y0), int(bbox[2]), int(bbox[3]))
                    if self.supervised_tracking_area is not None:
                        if bbox[0] < self.supervised_tracking_area[0]:
                            status = "Feature outside the specified area"
                        elif bbox[1] < self.supervised_tracking_area[1]:
                            status = "Feature outside the specified area"
                        elif bbox[0] + bbox[2] > self.supervised_tracking_area[0] + self.supervised_tracking_area[2]:
                            status = "Feature outside the specified area"
                        elif bbox[1] + bbox[3] > self.supervised_tracking_area[1] + self.supervised_tracking_area[3]:
                            status = "Feature outside the specified area"

                self.queue_out.put((status, bbox))

