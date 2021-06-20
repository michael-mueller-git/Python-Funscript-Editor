""" Funscript Visualizer """

import time
import cv2
import logging

import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtGui, QtCore ,QtWidgets
from typing import Callable
from weakref import ref
from funscript_editor.utils.config import UI_CONFIG
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from funscript_editor.data.funscript import Funscript


class FunscriptVisualizer(QtCore.QThread):
    """ Funscript Visualizer Thread

    Args:
        preview_pane (QLabel): the visualizer preview pane
        select_point_callback (Callable[[int], None]): select point callback handler
    """

    def __init__(self,
            preview_pane :QtWidgets.QLabel,
            select_point_callback :Callable[[int], None] = None):
        QtCore.QThread.__init__(self)
        self.preview_pane = preview_pane
        self.select_point_callback = select_point_callback
        self.figure = Figure()
        self.preview_pane.canvas = FigureCanvas(self.figure)
        self.preview_pane.canvas.mpl_connect('pick_event', self.__onpick)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.preview_pane.canvas)
        self.preview_pane.setLayout(layout)
        self.ax = self.figure.add_subplot(1,1,1)
        self.ax.get_yaxis().set_visible(True)
        self.ax.get_xaxis().set_visible(False)
        self.figure.subplots_adjust(
                left=UI_CONFIG['visualizer']['left'],
                right=UI_CONFIG['visualizer']['right'],
                top=UI_CONFIG['visualizer']['top'],
                bottom=UI_CONFIG['visualizer']['bottom']
            )
        self.last_frame = -1
        self.next_frame = 0
        self.funscript = None
        self.update_data_flag = False
        self.quit = False


    __logger = logging.getLogger(__name__)


    def __del__(self):
        self.quit = True


    def set_select_point_callback(self, callback :Callable[[int], None]) -> None:
        """ Set the selcet point callback handler

        Args:
            callback (Callable[[int], None]): select point callback handler
        """
        self.select_point_callback = callback


    def __onpick(self, event) -> None:
        """ onpick event handler for the matplotlib figure """
        if isinstance(event.artist, Line2D):
            x = np.take(event.artist.get_xdata(), event.ind)[0]
            y = np.take(event.artist.get_ydata(), event.ind)[0]
            if self.select_point_callback is not None: self.select_point_callback(x)


    def set_funscript(self, funscript: Funscript) -> None:
        """ Set the funscript to visualize

        Args
            funscript (Funscript): the funscript object
        """
        self.funscript = funscript
        self.__update_data()


    def update(self) -> None:
        """ Update the visualizer """
        self.__update_data()


    def __update_data(self) -> None:
        self.update_data_flag = True
        self.ax.clear()
        self.last_frame = -1
        if self.funscript is None: return
        self.ax.plot(self.funscript.get_actions_frames(), self.funscript.get_actions_positions())
        self.ax.plot(self.funscript.get_actions_frames(), self.funscript.get_actions_positions(), 'o', picker=5)
        self.ax.plot([self.next_frame, self.next_frame], [0,100])
        self.update_data_flag = False


    def set_frame(self, frame_number :int) -> None:
        """ Set current frame number

        Args:
            frame_number (int): current frame number
        """
        self.next_frame = frame_number


    def run(self) -> None:
        """ Funscript Visualizer Thread Function """
        while not self.quit:
            try:
                if self.last_frame == self.next_frame or self.funscript is None or self.update_data_flag:
                    time.sleep(0.001)
                    continue

                if len(self.ax.lines) > 1:
                    wr = ref(self.ax.lines[-1])
                    self.ax.lines.remove(wr())

                frame = max((0, self.next_frame))
                self.ax.plot([frame, frame], [0,100], 'k')
                self.ax.set_xlim([frame-self.funscript.get_fps(), frame+self.funscript.get_fps()])
                self.last_frame = frame
                self.preview_pane.canvas.draw()
            except: pass

