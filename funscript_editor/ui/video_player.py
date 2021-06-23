""" Video Player Pane """

import locale
import cv2
import time
import os
import logging
import mpv # pip install python-mpv

from typing import Callable
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtGui, QtCore ,QtWidgets
from funscript_editor.definitions import CONFIG_DIR
from funscript_editor.utils.config import UI_CONFIG
from funscript_editor.data.funscript import Funscript

import numpy as np


class VideoPlayer(QtCore.QThread):
    """ Video Player Thread

    Args:
        video_pane (QLabel): the visualizer preview pane
        frame_changed_callback (Callable[[int], None]): frame changed handler
        key_callback (Callable[[str], None]): key pressed (in video pane) handler
    """

    def __init__(self,
            video_pane :QtWidgets.QLabel,
            frame_changed_callback  :Callable[[int], None] = None,
            key_callback  :Callable[[str], None] = None):
        QtCore.QThread.__init__(self)
        self.font = ImageFont.truetype(os.path.join(CONFIG_DIR, "font.ttf"), UI_CONFIG['player']['fontsize'])
        self.video_pane = video_pane
        self.frame_changed_callback = frame_changed_callback
        locale.setlocale(locale.LC_NUMERIC, 'C')
        self.video_pane.setAttribute(QtCore.Qt.WA_DontCreateNativeAncestors)
        self.video_pane.setAttribute(QtCore.Qt.WA_NativeWindow)
        self.player = mpv.MPV(wid=str(int(self.video_pane.winId())),
                log_handler=self.__log_handler,
                input_default_bindings=True,
                input_vo_keyboard=True,
                osc=True,
                script_opts='osc-layout=box,osc-seekbarstyle=bar,osc-deadzonesize=0,osc-minmousemove=3',
                loglevel='error')
        self.player.loop_playlist = 'inf'
        self.overlay = self.player.create_image_overlay()
        self.last_timestamp = -1.0
        self.video_file = None
        self.indicate_bussy = False
        self.bussy_indicator_pos = 0
        self.funscript = None
        self.fps = 60
        self.stroke_indicator_position = UI_CONFIG['player']['stroke_indicator_pos']
        self.stroke_indicator_size = UI_CONFIG['player']['stroke_indicator_size']
        self.quit = False
        self.duration = 0
        self.length = 0
        self.key_callback = key_callback
        self.player.pause = True
        self.invert_stroke_indicator = False

        @self.player.property_observer('time-pos')
        def __time_observer(_name, value):
            if value is None: return
            if self.frame_changed_callback is not None:
                self.frame_changed_callback(self.millisec_to_frame(value*1000))

        @self.player.on_key_press('q')
        def __q_binding():
            pass

        @self.player.on_key_press('ctrl+g')
        def __ctrl_g_binding():
            if self.key_callback is not None:
                self.key_callback('ctrl+g')

        @self.player.on_key_press('ctrl+right')
        def __ctrl_right_binding():
            if self.key_callback is not None:
                self.key_callback('ctrl+right')

        @self.player.on_key_press('ctrl+shift+right')
        def __ctrl_shift_right_binding():
            if self.key_callback is not None:
                self.key_callback('ctrl+shift+right')

        @self.player.on_key_press('ctrl+left')
        def __ctrl_left_binding():
            if self.key_callback is not None:
                self.key_callback('ctrl+left')

        @self.player.on_key_press('ctrl+shift+left')
        def __ctrl_shift_left_binding():
            if self.key_callback is not None:
                self.key_callback('ctrl+shift+left')

        @self.player.on_key_press('0')
        def __ctrl_0_binding():
            if self.key_callback is not None:
                self.key_callback('0')

        @self.player.on_key_press('1')
        def __ctrl_1_binding():
            if self.key_callback is not None:
                self.key_callback('1')

        @self.player.on_key_press('2')
        def __ctrl_2_binding():
            if self.key_callback is not None:
                self.key_callback('2')

        @self.player.on_key_press('3')
        def __ctrl_3_binding():
            if self.key_callback is not None:
                self.key_callback('3')

        @self.player.on_key_press('4')
        def __ctrl_4_binding():
            if self.key_callback is not None:
                self.key_callback('4')

        @self.player.on_key_press('5')
        def __ctrl_5_binding():
            if self.key_callback is not None:
                self.key_callback('5')

        @self.player.on_key_press('6')
        def __ctrl_6_binding():
            if self.key_callback is not None:
                self.key_callback('6')

        @self.player.on_key_press('7')
        def __ctrl_7_binding():
            if self.key_callback is not None:
                self.key_callback('7')

        @self.player.on_key_press('8')
        def __ctrl_8_binding():
            if self.key_callback is not None:
                self.key_callback('8')

        @self.player.on_key_press('9')
        def __ctrl_9_binding():
            if self.key_callback is not None:
                self.key_callback('9')

        @self.player.on_key_press('del')
        def __delete_binding():
            if self.key_callback is not None:
                self.key_callback('delete')

        @self.player.on_key_press('shift+end')
        def __shift_end_binding():
            if self.key_callback is not None:
                self.key_callback('shift+end')

        @self.player.on_key_press('shift+home')
        def __shift_home_binding():
            if self.key_callback is not None:
                self.key_callback('shift+home')

        @self.player.on_key_press('end')
        def __end_binding():
            self.seek_time(self.get_duration)

        @self.player.on_key_press('home')
        def __home_binding():
            self.seek_time(0)

        @self.player.on_key_press('ctrl+s')
        def __ctrl_s_binding():
            if self.key_callback is not None:
                self.key_callback('ctrl+s')

        @self.player.on_key_press('ctrl++')
        def __ctrl_plus_binding():
            self.inc_stroke_indicator()

        @self.player.on_key_press('ctrl+-')
        def __ctrl_minus_binding():
            self.dec_stroke_indicator()

        @self.player.on_key_press('ctrl+i')
        def __ctrl_i_binding():
            self.toogle_stroke_indicator_inversion()

        @self.player.on_key_press('w')
        def __w_binding():
            self.move_stroke_indicator_relative(0,-1)

        @self.player.on_key_press('s')
        def __s_binding():
            self.move_stroke_indicator_relative(0,+1)

        @self.player.on_key_press('a')
        def __a_binding():
            self.move_stroke_indicator_relative(-1,0)

        @self.player.on_key_press('d')
        def __d_binding():
            self.move_stroke_indicator_relative(+1,0)


    __logger = logging.getLogger(__name__)


    def __del__(self):
        self.quit = True


    def __log_handler(self,a,b,c):
        pass


    def toogle_stroke_indicator_inversion(self) -> None:
        """ Toggle the stroke indicator inversion """
        self.invert_stroke_indicator = not self.invert_stroke_indicator


    @property
    def get_duration(self) -> float:
        """ Get current Video duration in seconds

        Returns:
            float: video duration in secondes
        """
        return self.duration


    @property
    def get_fps(self) -> float:
        """ Get current Video FPS

        Returns:
            float: Video FPS
        """
        return self.fps


    @property
    def get_video_file(self) -> str:
        """ Get current video file path

        Returns:
            str: video file path
        """
        return self.video_file


    @property
    def get_current_timestamp_in_millis(self) -> int:
        """ Get current video timestamp in milliseconds

        Returns:
            int: video timestamp in milliseconds
        """
        ts = self.player.time_pos
        if ts is None: return 0
        return int(ts * 1000)


    @property
    def get_length(self) -> int:
        """ Get video length in frames

        Returns:
            int: number of frames in video
        """
        return self.length


    @property
    def get_current_frame(self) -> int:
        """ Get current frame number

        Returns:
            int: current frame number
        """
        return self.millisec_to_frame(self.get_current_timestamp_in_millis)


    def increase_speed(self) -> None:
        """ Increase the playback speed """
        self.player.speed += 0.2
        if 0.9 < self.player.speed < 1.1: self.player.speed = 1.0
        self.show_message('Speed: ' + str(self.player.speed))


    def decrease_speed(self) -> None:
        """ Decrease the playback speed """
        self.player.speed -= 0.2
        if 0.9 < self.player.speed < 1.1: self.player.speed = 1.0
        self.show_message('Speed: ' + str(self.player.speed))


    def show_message(self, message :str) -> None:
        """ Show a message on the video pane

        Args:
            message (str): the message to display
        """
        self.player.command('show-text', message)


    def toggle_play_pause(self) -> None:
        """ Toggle Play/Pause """
        self.player.pause = not self.player.pause


    def set_stroke_indicator_position(self, x :int, y :int) -> None:
        """ Set the absolute stroke indicator position in percent

        Args:
            x (int): x position in percent [1, 99]
            y (int): y position in percent [1, 99]
        """
        self.stroke_indicator_position = [min((max((x, 1)), 99)),min((max((y, 1)), 99))]
        self.last_timestamp = -1.0


    def move_stroke_indicator_relative(self, x, y) -> None:
        """ Move the stroke indicator relative in perecent

        Args:
            x (int): x move in percent [-99, +99]
            y (int): y move in percent [-99, +99]
        """
        self.set_stroke_indicator_position(
                self.stroke_indicator_position[0]+x,
                self.stroke_indicator_position[1]+y
            )


    def set_stroke_indicator_size(self, size):
        """ Set the absolute stroke indicator size in percent

        Args:
            size (int): stroke indicator size in percent [1, 99]
        """
        self.stroke_indicator_size = size
        self.last_timestamp = -1.0


    def inc_stroke_indicator(self) -> None:
        """ Increase the stroke indicator size """
        self.stroke_indicator_size = min((99, self.stroke_indicator_size+1))
        self.last_timestamp = -1.0


    def dec_stroke_indicator(self):
        """ Decrease the stroke indicator size """
        self.stroke_indicator_size = max((1, self.stroke_indicator_size-1))
        self.last_timestamp = -1.0


    def set_indicate_bussy(self, value: bool) -> None:
        """ Enable/Disable the bussy indicator on the video pane

        Args:
            value (bool): True to show the bussy indicator, False to hide the bussy indicator
        """
        self.indicate_bussy = value


    def set_frame_changed_callback(self, callback :Callable[[int], None]) -> None:
        """ Set the frame changed handler

        Args:
            callback (Callable[[int], None]): frame changed callback handler
        """
        self.frame_changed_callback = callback


    def open(self, video_file :str) -> None:
        """ Open a new video file

        Args:
            video_file (str): path to video file
        """
        self.player.pause = True
        cap = cv2.VideoCapture(video_file)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.player.loadfile(video_file)
        self.video_file = video_file


    def get_screenshot(self) -> np.ndarray:
        """ Get a screenshot of current video position

        Returns:
            np.ndarray: opencv image
        """
        try: return cv2.cvtColor(np.array(self.player.screenshot_raw()), cv2.COLOR_RGB2BGR)
        except: return None


    def set_funscript(self, funscript :Funscript) -> None:
        """ Set the funscript for the stroke indicator visualizer

        Args:
            funscript (Funscript): the funscript object
        """
        self.funscript = funscript


    def seek_frame(self, frame_number :int) -> None:
        """ Seek Video player to given frame number

        Args:
            frame_number (int): frame number
        """
        if self.length < 1: return
        self.player.seek(self.frame_to_millisec(frame_number) * 1.0 / 1000, "absolute", "exact")


    def seek_time(self, timestamp_in_seconds: float) -> None:
        """ Seek Video player to given timestamp

        Args:
            timestamp_in_seconds (float): timestamp in seconds
        """
        self.player.seek(timestamp_in_seconds, "absolute", "exact")


    def show_prev_frame(self) -> None:
        """ Show previous video frame """
        if self.video_file is None: return
        ts = self.player.time_pos
        if ts is None: return
        ts -= 1.0/float(self.fps)
        if ts > 0: self.seek_time(ts)


    def show_next_frame(self) -> None:
        """ Show next video frame """
        if self.video_file is None: return
        ts = self.player.time_pos
        ts += 1.0/float(self.fps)
        self.seek_time(ts)


    def millisec_to_frame(self, milliseconds :int) -> int:
        """ Convert a timestamp in milliseconds to frame number for current video

        Args:
            milliseconds (int): timestamp in milliseconds

        Returns:
            int: fram number for current video
        """
        if self.fps is None or self.fps == 0: return 0
        if milliseconds < 0: return 0
        return int(round(float(milliseconds)/(float(1000)/float(self.fps))))


    def frame_to_millisec(self, frame :int) -> int:
        """ Convert the frame number to an timestamp in milliseconds for current video

        Args:
            frame (int): frame number

        Returns:
            int: timestamp in milliseconds
        """
        if frame < 0: return 0
        return int(round(float(frame)*float(1000)/self.fps))


    def __update_bussy_indicator(self) -> None:
        img = Image.new('RGBA', (255, 255),  (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        arc_start = 360 - (self.bussy_indicator_pos % 360)
        arc_end = 360 - ((self.bussy_indicator_pos+270) % 360)
        self.bussy_indicator_pos -= 16
        d.arc((0, 0, 127, 127), start=arc_start, end=arc_end, fill="red", width=12)
        self.overlay.update(img, pos=(int(self.video_pane.size().width()/2-64),
            int(self.video_pane.size().height()/2-64)))
        self.last_timestamp = -1.0


    def __update_stroke_indicator(self) -> None:
        ts = self.player.time_pos
        if ts is None: return
        if ts == self.last_timestamp: return

        self.last_timestamp = ts

        img = Image.new('RGBA', (25, 125),  (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        if self.funscript is not None and not self.funscript.is_empty():
            pos = 100 - np.interp(
                    ts,
                    np.array(self.funscript.get_actions_times()) / float(1000),
                    self.funscript.get_actions_positions()
                )
            d.rectangle([(0, 0), (20, 100)], outline ="red", width=3)

            if self.invert_stroke_indicator:
                d.rectangle([(0, 100), (20, pos)], fill='red', outline ="red", width=3)
            else:
                d.rectangle([(0, 100), (20, 100 - pos)], fill='red', outline ="red", width=3)

            h = round(self.video_pane.size().height() / float(100) * self.stroke_indicator_size)
            w = round(h/float(100)*20)
            img = img.resize((w,h))

        self.overlay.update(
                img,
                pos=(
                    int(self.video_pane.size().width()/100 * self.stroke_indicator_position[0]),
                    int(self.video_pane.size().height()/100 * self.stroke_indicator_position[1])
                    )
                )


    def send_mouse_pos(self, x :int, y :int) -> None:
        """ Control mouse on mpv via commands

        Args:
            x (int): x position
            y (int): y position
        """
        self.player.command("mouse", str(x), str(y))
        self.player.command("keydown", "MOUSE_BTN0")
        time .sleep(0.01)
        self.player.command("keyup", "MOUSE_BTN0")
        # print("[DEBUG] click ui at", x, y)


    def run(self) -> None:
        """ Video Player Thread Function """
        self.player.wait_until_playing()
        while not self.quit:
            try:
                time.sleep(0.01)
                if self.indicate_bussy: self.__update_bussy_indicator()
                else: self.__update_stroke_indicator()
            except: pass
        self.__logger.info("quit video player")
        self.player.terminate()



