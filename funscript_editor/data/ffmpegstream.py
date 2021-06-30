""" FFmpeg Video Stream """

import cv2
import os
import logging
import time

from dataclasses import dataclass
from threading import Thread
from queue import Queue

import subprocess as sp
import numpy as np


@dataclass
class FFmpegStreamParameter:
    """ FFmpeg Stream Parameter Dataclass with default values """
    fov: int = 100
    theta: int = -90
    phi: int = -45
    height: int = 720
    width: int = 1240


@dataclass
class VideoInfo:
    """ Video Info Dataclass """
    fps :float
    height :int
    width :int
    length :int


class FFmpegStream:
    """ FFmpeg Stream with frame buffer

    Args:
        video_path (str): path to video file
        parameter (FFmpegStreamParameter): conversion parameter
        start_frame (int): start frame number
        queue_size (int): size of frame buffer
    """

    def __init__(self,
            video_path :str,
            parameter :FFmpegStreamParameter,
            start_frame :int = 0,
            queue_size :int = 256):

        self.video_path = video_path
        self.parameter = parameter
        self.start_frame = start_frame
        self.queue_size = queue_size

        self.stopped = False
        self.current_frame = 0
        self.sleep_time = 0.001

        self.video_info = self.get_video_info(video_path)
        self.frame_buffer = Queue(maxsize=queue_size)

        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()


    logger = logging.getLogger(__name__)


    @staticmethod
    def get_video_info(
            video_path: str) -> VideoInfo:
        """ Get VideoInfo

        Args:
            video_path (str): path to video

        Returns:
            VideoInfo: video infos
        """
        cap = cv2.VideoCapture(video_path)
        video_info = VideoInfo(
                fps = float(cap.get(cv2.CAP_PROP_FPS)),
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )
        cap.release()
        return video_info


    @staticmethod
    def get_frame(
            video_path :str,
            frame_number :int) -> np.ndarray:
        """ Get Video frame

        Args:
            video_path (str): path to video
            frame_number (int): frame number to extract from video

        Returns:
            np.ndarray: opencv image
        """
        cap = cv2.VideoCapture(video_path)
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        cap.release()
        return frame


    @staticmethod
    def get_projection(
            frame :np.ndarray,
            parameter: FFmpegStreamParameter) -> np.ndarray:
        """ Get projection of frame

        Args:
            frame (np.ndarray): opencv image
            parameter (FFmpegStreamParameter): conversion parameter

        Returns:
            np.ndarray: projected opencv image
        """
        dimension = '{}x{}'.format(frame.shape[1], frame.shape[0])

        video_filter = 'v360=input=he' \
                        + ':in_stereo=sbs' \
                        + ':pitch=' + str(parameter.phi) \
                        + ':output=flat' \
                        + ':d_fov=' + str(parameter.fov) \
                        + ':w=' + str(parameter.width) \
                        + ':h=' + str(parameter.height)

        command = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'warning',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', dimension,
                '-pix_fmt', 'bgr24',
                '-i', '-',
                '-f', 'image2pipe',
                '-pix_fmt', 'bgr24',
                '-vsync', '0',
                '-vcodec', 'rawvideo',
                '-an',
                '-sn',
                '-vf', video_filter,
                '-'
            ]

        pipe = sp.Popen(
                command,
                stdin = sp.PIPE,
                stdout = sp.PIPE,
                bufsize = 3 * parameter.width * parameter.height
            )

        pipe.stdin.write(frame.tobytes())
        projection = np.frombuffer(
                pipe.stdout.read(parameter.width * parameter.height * 3),
                dtype='uint8'
            ).reshape(
                    (parameter.height, parameter.width, 3)
                )

        pipe.stdin.close()
        pipe.stdout.close()
        pipe.terminate()

        return projection


    def frame_to_millisec(self, frame_number) -> int:
        """Get timestamp for given frame number

        Args:
            frame_number (int): frame number

        Returns:
            int: timestamp in video
        """
        if frame_number <= 0: return 0
        return int(round(float(frame_number)*float(1000)/self.video_info.fps))


    def millisec_to_timestamp(self, millis :int)->str:
        """ Convert milliseconds to timestamp

        Args:
            millis (int): position in video in milliseconds

        Returns:
            str: position in video as timestamp with H:M:S.XXX
        """
        millis = int(millis)
        seconds = int((millis / 1000) % 60)
        minutes = int((millis / (1000 * 60)) % 60)
        hours = int((millis / (1000 * 60 * 60)) % 24)
        millis = int(millis % 1000)

        return str(hours).zfill(2) \
                + ':' + str(minutes).zfill(2) \
                + ':' + str(seconds).zfill(2) \
                + '.' + str(millis).zfill(3)


    def stop(self) -> None:
        """ Stop FFmpeg video stream """
        self.stopped = True
        self.thread.join()


    def read(self) -> np.ndarray:
        """ Get next projected frame from stream

        Returns:
            np.ndarray: opencv image data
        """
        return self.frame_buffer.get()


    def isOpen(self) -> bool:
        """ Check if FFmpeg video stream is open or a frame is still available in the buffer

        Returns:
            bool: True if video stream is open or a frame is still available in the buffer else False
        """
        return self.more() or not self.stopped


    def more(self) -> bool:
        """ Check if frames in the frame bufer are available

        Returns:
            bool: True if a frame is available else False
        """
        tries = 0
        while self.frame_buffer.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(self.sleep_time)
            tries += 1

        return self.frame_buffer.qsize() > 0


    def run(self) -> None:
        """ Function to read transformed frames from ffmpeg video stream into a queue """

        video_filter = 'v360=input=he' \
                        + ':in_stereo=sbs' \
                        + ':pitch=' + str(self.parameter.phi) \
                        + ':output=flat' \
                        + ':d_fov=' + str(self.parameter.fov) \
                        + ':w=' + str(self.parameter.width) \
                        + ':h=' + str(self.parameter.height)

        seek = self.millisec_to_timestamp(
                self.frame_to_millisec(self.start_frame)
            )

        command = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'warning',
                '-ss', str(seek),
                '-i', self.video_path,
                '-f', 'image2pipe',
                '-pix_fmt', 'bgr24',
                '-vsync', '0',
                '-vcodec', 'rawvideo',
                '-an',
                '-sn',
                '-vf', video_filter,
                '-'
            ]

        pipe = sp.Popen(
                command,
                stdout = sp.PIPE,
                bufsize= 3 * self.parameter.height * self.parameter.width
            )

        while not self.stopped:
            data = pipe.stdout.read(self.parameter.width * self.parameter.height * 3)
            if not data:
                break

            frame = np.frombuffer(data, dtype='uint8').reshape(
                    (self.parameter.height, self.parameter.width, 3)
                )
            if frame is None:
                break

            while self.frame_buffer.full() and not self.stopped:
                time.sleep(self.sleep_time)

            self.frame_buffer.put(frame)
            self.current_frame += 1

        self.logger.info('Close FFmpeg Stream')
        pipe.stdout.close()
        pipe.terminate()
        self.stopped = True
