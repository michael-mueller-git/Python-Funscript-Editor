import cv2
import os
import copy
import threading
import pynput.keyboard

from playsound import playsound
from queue import Queue
from datetime import datetime
from funscript_editor.data.ffmpegstream import FFmpegStream, VideoInfo
from funscript_editor.utils.config import PROJECTION, SETTINGS, NOTIFICATION_SOUND_FILE
from screeninfo import get_monitors
from dataclasses import dataclass

import funscript_editor.utils.logging as logging
import numpy as np


@dataclass
class OpenCV_GUI_Parameters:
    video_info: VideoInfo
    skip_frames: int
    end_frame_number: int
    preview_scaling: float = float(SETTINGS['preview_scaling'])
    text_start_x: int = 30
    text_start_y: int = 30
    text_line_height: int = 30
    font_size: float = 0.75
    fps_smoothing_factor: int = 100
    window_name_prefix: str = "MTFG"
    notification_sound_file = NOTIFICATION_SOUND_FILE
    use_zoom: bool = SETTINGS["use_zoom"]
    zoom_factor: float = max((1.0, float(SETTINGS['zoom_factor'])))
    text_border_width: int = 6


class KeypressHandler:
    """ Keypress Handler for OpenCV GUI """

    def __init__(self):
        self.keypress_queue = Queue(maxsize=32)
        self.listener = pynput.keyboard.Listener(
            on_press = self.on_key_press,
            on_release = None
        )
        self.listener.start()


    def __del__(self):
        self.listener.stop()


    def on_key_press(self, key: pynput.keyboard.Key) -> None:
        """ Our key press handle to register the key presses

        Args:
            key (pynput.keyboard.Key): the pressed key
        """
        if not self.keypress_queue.full():
            # print('{0}'.format(key))
            self.keypress_queue.put(key)


    def clear_keypress_queue(self) -> None:
        """ Clear the key press queue """
        while self.keypress_queue.qsize() > 0:
            self.keypress_queue.get()


    def was_any_accept_key_pressed(self) -> bool:
        """ Check if 'space' or 'enter' was presssed

        Returns:
            bool: True if an accept key was pressed else False
        """
        while self.keypress_queue.qsize() > 0:
            if any('{0}'.format(self.keypress_queue.get()) == x for x in ["Key.space", "Key.enter"]):
                return True

        return False


    def was_key_pressed(self, key: str) -> bool:
        """ Check if key was presssed

        Args:
            key (str): the key to check

        Returns:
            bool: True if 'q' was pressed else False
        """
        if key is None or len(key) == 0:
            return False

        while self.keypress_queue.qsize() > 0:
            if '{0}'.format(self.keypress_queue.get()) == "'"+key[0]+"'":
                return True

        return False



class OpenCV_GUI(KeypressHandler):
    """ High Level OpenCV GUI.

    Args:
        params (OpenCV_GUI_Parameters): configuration parameters for OpenCV GUI
    """

    def __init__(self, params: OpenCV_GUI_Parameters):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.preview_fps = []
        self.fps_timer = cv2.getTickCount()
        self.preview_image_origin_height = 0
        self.preview_image_origin_width = 0
        self.window_name = "{} - {}".format(self.params.window_name_prefix, datetime.now().strftime("%H:%M:%S"))
        self.__reset_print_positions()
        self.preview_scaling_applied = False
        self.preview_image = None
        self.preview_image_without_scale = None


    def __del__(self):
        super().__del__()
        self.close()


    def close(self):
        """Close all OpenCV GUIs"""
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass


    def __reset_print_positions(self):
        """ Reset all print positions """
        self.text_y_pos = {
                'left': self.params.text_start_y,
                'center': self.params.text_start_y,
                'column2': self.params.text_start_y,
                'right': self.params.text_start_y
            }


    def __determine_preview_scaling(self) -> None:
        """ Determine the scaling for current monitor setup """
        scale = []
        try:
            for monitor in get_monitors():
                if monitor.width > monitor.height:
                    scale.append(
                        min([
                            monitor.width / float(self.preview_image_origin_width),
                            monitor.height / float(self.preview_image_origin_height)
                        ])
                    )
        except:
            pass

        if len(scale) == 0:
            self.logger.error("Monitor resolution info not found")
            self.monitor_preview_scaling = 1.0
        else:
            # asume we use the largest monitor for scipting
            self.monitor_preview_scaling = self.params.preview_scaling * max(scale)
            self.monitor_preview_scaling = self.monitor_preview_scaling / float(os.getenv("QT_SCALE_FACTOR", 1))


    def set_background_image(self, image: np.ndarray, copy_image: bool = False) -> None:
        """ Set the preview image

        Args:
            image (np.ndarray): opencv image
            copy_image (bool): create an copy of the image
        """
        if image is None:
            image = np.full((720, 1240, 3), 0, dtype=np.uint8)

        self.preview_image = copy.deepcopy(image) if copy_image else image

        if self.preview_image.shape[0] != self.preview_image_origin_height or self.preview_image.shape[1] != self.preview_image_origin_width:
            self.preview_image_origin_height = self.preview_image.shape[0]
            self.preview_image_origin_width = self.preview_image.shape[1]
            self.__determine_preview_scaling()

        self.preview_scaling_applied = False
        self.__reset_print_positions()


    def draw_box(self, bbox, color: tuple = (255, 0, 255)) -> None:
        """ Draw an tracking box to the preview image

        Args:
            bbox (tuple): tracking box with (x,y,w,h)
            color (tuple): RGB color values for the box
        """
        assert self.preview_image is not None

        if not isinstance(bbox, list):
            bbox = [bbox]

        for box in bbox:
            if box and len(box) >= 4:
                cv2.rectangle(
                        self.preview_image,
                        (box[0], box[1]),
                        ((box[0] + box[2]), (box[1] + box[3])),
                        color,
                        3,
                        1
                    )
                if len(box) >= 6:
                    cv2.circle(
                            self.preview_image,
                            (box[4], box[5]),
                            4,
                            color,
                            2
                        )


    @staticmethod
    def draw_box_to_image(image: np.ndarray, bbox, color: tuple = (255, 0, 255)) -> np.ndarray:
        """ Draw an tracking box to given image

        Args:
            image (np.ndarray): image
            bbox (tuple): tracking box with (x,y,w,h)
            color (tuple): RGB color values for the box

        Returns:
            np.ndarray: opencv image
        """
        if not isinstance(bbox, list):
            bbox = [bbox]

        for box in bbox:
            if box and len(box) >= 4:
                cv2.rectangle(
                        image,
                        (box[0], box[1]),
                        ((box[0] + box[2]), (box[1] + box[3])),
                        color,
                        3,
                        1
                    )
                if len(box) >= 6:
                    cv2.circle(
                            image,
                            (box[4], box[5]),
                            4,
                            color,
                            2
                        )

        return image


    def get_preview_fps(self) -> float:
        """ Get current processing FPS

        Returns
            float: FPS
        """
        if len(self.preview_fps) < 1:
            return 1.0

        return np.mean((
            self.preview_fps[-self.params.fps_smoothing_factor:] \
            if len(self.preview_fps) < self.params.fps_smoothing_factor \
            else self.preview_fps
        ))


    def print_fps(self) -> None:
        """ Draw processing FPS to the preview image """
        assert self.preview_image is not None
        self.print_text(str(int(self.get_preview_fps())) + ' fps')


    def __update_processing_fps(self) -> None:
        """ Update processing FPS """
        self.preview_fps.append((self.params.skip_frames+1) * cv2.getTickFrequency() / (cv2.getTickCount()-self.fps_timer))
        self.fps_timer = cv2.getTickCount()


    def print_time(self, current_frame_number: int) -> None:
        """ Draw Time on the preview image

        Args:
            current_frame_number (int): current absolute frame number
        """
        assert self.preview_image is not None
        current_timestamp = FFmpegStream.frame_to_timestamp(current_frame_number, self.params.video_info.fps)
        current_timestamp = ''.join(current_timestamp[:-4])

        if self.params.end_frame_number < 1:
            end_timestamp = FFmpegStream.frame_to_timestamp(self.params.video_info.length, self.params.video_info.fps)
            end_timestamp = ''.join(end_timestamp[:-4])
        else:
            end_timestamp = FFmpegStream.frame_to_timestamp(self.params.end_frame_number, self.params.video_info.fps)
            end_timestamp = ''.join(end_timestamp[:-4])

        txt = current_timestamp + ' / ' + end_timestamp
        self.print_text(txt, text_position_x = 'right')


    def print_text(self, txt, color: tuple = (0,0,255), text_position_x: str = 'left') -> None:
        """ Draw text to an image/frame

        Args:
            txt (str, list): the text to plot on the image
            colot (tuple): BGR Color tuple
            text_position_x (str): text position ['left',  'right', 'center', 'column2']
        """
        assert self.preview_image is not None
        assert text_position_x in self.text_y_pos.keys()

        if not isinstance(txt, list):
            txt = [txt]

        for line in txt:
            if text_position_x.lower() == 'left':
                x = self.params.text_start_x
            elif text_position_x.lower() == 'right':
                (text_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.params.font_size, 2)
                x = max([0, int(self.preview_image.shape[1] - self.params.text_start_x - text_w) ])
            elif text_position_x.lower() == 'center':
                (text_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.params.font_size, 2)
                x = max([0, round((self.preview_image_origin_width / 2) - (text_w / 2))])
            elif text_position_x.lower() == 'column2':
                x = round(self.preview_image_origin_width / 2 + self.params.text_start_x)
            else:
                raise NotImplementedError("Print Text at position %s is not implemented", text_position_x)

            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.params.font_size, 2)
            cv2.rectangle(
                    self.preview_image,
                    (x - self.params.text_border_width, self.text_y_pos[text_position_x] - self.params.text_border_width),
                    (x + text_w + self.params.text_border_width, self.text_y_pos[text_position_x] + text_h + self.params.text_border_width),
                    (0, 0, 0),
                    -1
                )
            cv2.putText(
                    self.preview_image,
                    str(line),
                    (x, round(self.text_y_pos[text_position_x] + text_h + self.params.font_size - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.params.font_size,
                    color,
                    2
                )

            self.text_y_pos[text_position_x] += self.params.text_line_height


    def scale_preview_image(self)-> None:
        """ Scale image for preview """
        assert self.preview_image is not None

        if self.preview_scaling_applied:
            return

        self.preview_image_without_scale = copy.deepcopy(self.preview_image)
        self.preview_image = cv2.resize(
                self.preview_image,
                None,
                fx=self.monitor_preview_scaling,
                fy=self.monitor_preview_scaling
            )

        self.preview_scaling_applied = True


    def show(self, wait: int = 1) -> int:
        """ Show annotated preview image

        Args:
            wait (int): waitKey time in milliseconds

        Returns:
            int: waitKey result
        """
        assert self.preview_image is not None
        self.scale_preview_image()
        cv2.imshow(self.window_name, self.preview_image)
        return cv2.waitKey(wait)


    def selectROI(self) -> tuple:
        """ OpenCV selectROI wrapper

        Returns:
            tuple: bbox (x,y,w,h)
        """
        assert self.preview_image is not None
        self.scale_preview_image()
        return cv2.selectROI(self.window_name, self.preview_image, False)


    def show_loading_screen(self, txt: str = "Please wait...") -> None:
        """ Show an loading screen

        Args:
            txt (str): text to display
        """
        if self.preview_image_without_scale is not None:
            self.set_background_image(np.full(self.preview_image_without_scale.shape, 0, dtype=np.uint8))
            self.print_text(txt, color=(0,0,255))
            self.show()


    def play_notification_sound(self) -> None:
        """ Play an sound to signal an event """
        if self.params.notification_sound_file is not None:
            if os.path.exists(self.params.notification_sound_file):
                def play_beep():
                    try:
                        playsound(self.params.notification_sound_file)
                    except:
                        pass
                try:
                    self.beep_thread = threading.Thread(target=play_beep)
                    self.beep_thread.start()
                except:
                    pass
            else:
                self.logger.warning("Notification sound file not found (%s)", self.params.notification_sound_file)


    def min_max_selector(self,
            image_min: np.ndarray,
            image_max: np.ndarray,
            info: str = "",
            title_min: str = "",
            title_max: str = "",
            lower_limit: int = 0,
            upper_limit: int = 99,
            beep: bool = False) -> tuple:
        """ Min Max selection Window

        Args:
            image_min (np.ndarray): the frame/image with lowest position
            image_max (np.ndarray): the frame/image with highest position
            info (str): additional info string th show on the Window
            title_min (str): title for the min selection
            title_max (str): title for the max selection
            lower_limit (int): the lower possible value
            upper_limit (int): the highest possible value
            beep (bool): play notification sound

        Returns:
            tuple: with selected (min: flaot, max float)
        """
        cv2.createTrackbar("Min", self.window_name, lower_limit, upper_limit, lambda _: None)
        cv2.createTrackbar("Max", self.window_name, upper_limit, upper_limit, lambda _: None)
        image = np.concatenate((image_min, image_max), axis=1)

        if beep:
            self.play_notification_sound()

        self.clear_keypress_queue()
        trackbarValueMin = lower_limit
        trackbarValueMax = upper_limit
        self.logger.info("Waiting for user input")

        while True:
            try:
                self.set_background_image(image, copy_image=True)
                self.print_text(title_min if title_min != "" else "Min")
                self.print_text(title_max if title_max != "" else "Max", text_position_x='column2')
                self.print_text("Set {} to {}".format('Min', trackbarValueMin))
                self.print_text("Set {} to {}".format('Max', trackbarValueMax), text_position_x='column2')
                self.print_text("Info: " + info)
                self.print_text("Press 'space' or 'enter' to continue", text_position_x='column2')
                ret = self.show(25)

                if self.was_any_accept_key_pressed() or any(ret == x for x in [ord(' '), 13]):
                    break

                trackbarValueMin = cv2.getTrackbarPos("Min", self.window_name)
                trackbarValueMax = cv2.getTrackbarPos("Max", self.window_name)
            except:
                pass

        self.logger.info("Receive User Input")
        self.show_loading_screen()

        return (trackbarValueMin, trackbarValueMax) \
                if trackbarValueMin < trackbarValueMax \
                else (trackbarValueMax, trackbarValueMin)


    @staticmethod
    def get_center(box: tuple) -> tuple:
        """ Get the cencter point of an box

        Args:
            box (tuple): the predicted bounding box

        Returns:
            tuple (x,y) of the current point
        """
        return ( round(box[0] + box[2]/2), round(box[1] + box[3]/2) )


    def bbox_selector(self, image: np.ndarray, txt: str, add_center: bool = False) -> tuple:
        """ Window to get an bounding box from user input

        Args:
            image (np.ndarray): opencv image e.g. the first frame to determine the bounding box
            txt (str): additional text to display on the selection window
            add_center (bool): add center cordinates to the box

        Returns:
            tuple: user input bounding box tuple (x,y,w,h)
        """
        if self.params.use_zoom:
            self.set_background_image(image, copy_image=True)
            self.print_text("Select area with Mouse and Press 'space' or 'enter' to continue")
            self.print_text("Zoom selected area")
            while True:
                zoom_bbox = self.selectROI()

                if zoom_bbox is None or len(zoom_bbox) == 0:
                    continue

                if zoom_bbox[2] < 75 or zoom_bbox[3] < 75:
                    self.logger.error("The selected zoom area is to small")
                    continue

                break

            zoom_bbox = (
                    round(zoom_bbox[0]/self.monitor_preview_scaling),
                    round(zoom_bbox[1]/self.monitor_preview_scaling),
                    round(zoom_bbox[2]/self.monitor_preview_scaling),
                    round(zoom_bbox[3]/self.monitor_preview_scaling)
                )

            image = image[zoom_bbox[1]:zoom_bbox[1]+zoom_bbox[3], zoom_bbox[0]:zoom_bbox[0]+zoom_bbox[2]]
            image = cv2.resize(image, None, fx=self.params.zoom_factor, fy=self.params.zoom_factor)

        self.set_background_image(image, copy_image=True)
        self.print_text("Select area with Mouse and Press 'space' or 'enter' to continue")
        self.print_text(txt)

        while True:
            bbox = self.selectROI()

            if bbox is None or len(bbox) == 0:
                continue

            if bbox[0] == 0 or bbox[1] == 0 or bbox[2] < 9 or bbox[3] < 9:
                continue

            break

        # revert the preview scaling
        bbox = (
                round(bbox[0]/self.monitor_preview_scaling),
                round(bbox[1]/self.monitor_preview_scaling),
                round(bbox[2]/self.monitor_preview_scaling),
                round(bbox[3]/self.monitor_preview_scaling)
            )

        # revert the zoom
        if self.params.use_zoom:
            bbox = (
                    round(bbox[0]/self.params.zoom_factor + zoom_bbox[0]),
                    round(bbox[1]/self.params.zoom_factor + zoom_bbox[1]),
                    round(bbox[2]/self.params.zoom_factor),
                    round(bbox[3]/self.params.zoom_factor)
                )

        if add_center:
            center = self.get_center(bbox)
            bbox = (bbox[0], bbox[1], bbox[2], bbox[3], center[0], center[1])

        self.logger.info("User Input: %s", str(bbox))
        return bbox


    def get_video_projection_config(self, image :np.ndarray, projection: str) -> dict:
        """ Get the video projection config form user input

        Args:
            image (np.ndarray): opencv vr 180 or 360 image
            projection (str): projection key from config

        Returns:
            dict: projection config
        """
        assert projection in PROJECTION.keys()
        config = copy.deepcopy(PROJECTION[projection])

        h, w = image.shape[:2]
        if PROJECTION[projection]['parameter']['height'] == -1:
            scaling = config['parameter']['width'] / float(w)
            config['parameter']['height'] = round(h * scaling)
        elif PROJECTION[projection]['parameter']['width'] == -1:
            scaling = config['parameter']['height'] / float(h)
            config['parameter']['width'] = round(w * scaling)

        # NOTE: improve processing speed to make this menu more responsive
        if image.shape[0] > 6000 or image.shape[1] > 6000:
            image = cv2.resize(image, None, fx=0.25, fy=0.25)

        if image.shape[0] > 3000 or image.shape[1] > 3000:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

        ui_texte = {}
        if "keys" in config.keys():
            for param in config['keys'].keys():
                if param in config['parameter'].keys() and all(item in config["keys"][param].keys() for item in ["increase", "decrease"]):
                    ui_texte[param] = "Use '{}', '{}' to increase/decrease {} = ${{val}}".format(
                        config["keys"][param]["increase"],
                        config["keys"][param]["decrease"],
                        param
                    )

        self.clear_keypress_queue()

        self.logger.info("Show ROI Menu")
        if len(ui_texte) > 0:
            # we need an user input
            parameter_changed, selected = True, False
            while not selected:
                if parameter_changed:
                    parameter_changed = False
                    preview = FFmpegStream.get_projection(image, config)
                    self.set_background_image(preview)
                    self.print_text("Press 'space' or 'enter' to use current viewpoint")
                    self.print_text("Press '0' (NULL) to reset view")
                    final_ui_texte = [ui_texte[k].replace('${val}', str(config['parameter'][k])) for k in ui_texte.keys()]
                    self.print_text(final_ui_texte)

                ret = self.show()
                if ret in [ord(' '), 13]:
                    break

                while self.keypress_queue.qsize() > 0:
                    pressed_key = '{0}'.format(self.keypress_queue.get())
                    if pressed_key == "Key.space" or pressed_key == "Key.enter":
                        selected = True
                        break

                    if pressed_key == "'0'":
                        config = copy.deepcopy(PROJECTION[projection])
                        if config['parameter']['height'] == -1:
                            scaling = config['parameter']['width'] / float(w)
                            config['parameter']['height'] = round(h * scaling)
                        elif config['parameter']['width'] == -1:
                            scaling = config['parameter']['height'] / float(h)
                            config['parameter']['width'] = round(w * scaling)
                        parameter_changed = True
                        break

                    if "keys" not in config.keys():
                        break

                    for param in config['keys'].keys():
                        if param in config['parameter'].keys() and all(x in config["keys"][param].keys() for x in ["increase", "decrease"]):
                            if pressed_key == "'" + config["keys"][param]["increase"] + "'":
                                config['parameter'][param] += 5
                                parameter_changed = True
                                break
                            elif pressed_key == "'" + config["keys"][param]["decrease"] + "'":
                                config['parameter'][param] -= 5
                                parameter_changed = True
                                break


        self.show_loading_screen()
        return config



    def preview(self,
            image: np.ndarray,
            current_frame_number,
            texte: list = [],
            boxes: list = [],
            wait: int = 1,
            show_fps: bool = True,
            show_time: bool = True,
            beep: bool = False
            ) -> int:
        """
        Args:
            image (np.ndarray): image to preview
            current_frame_number (int): current frame number
            texte (list, optional): list of texte to annotate the preview image
            boxes (list, optional): draw boxes on the preview image
            wait (int): waitKey delay in milliseconds
            show_fps (bool): show processing fps
            show_time (bool): show processing time
            beep (bool): play notification sound

        Returns:
            int: waitKey result
        """
        self.set_background_image(image)
        self.__update_processing_fps()

        if show_time:
            self.print_time(current_frame_number)

        if show_fps:
            self.print_fps()

        self.print_text(texte)
        self.draw_box(boxes)

        if beep:
            self.play_notification_sound()

        return self.show(wait)
