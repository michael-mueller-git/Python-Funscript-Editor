""" Main Window """

import cv2
import time
import os
import copy
import locale
import threading
import logging
import shutil
import webbrowser
import platform

import numpy as np

from PyQt5 import QtGui, QtCore ,QtWidgets

from datetime import datetime

from funscript_editor.ui.funscript_editor_view import Ui_MainWindow
from funscript_editor.ui.video_player import VideoPlayer
from funscript_editor.ui.funscript_visualizer import FunscriptVisualizer
from funscript_editor.data.funscript import Funscript
from funscript_editor.algorithms.funscriptgenerator import FunscriptGenerator, FunscriptGeneratorParameter
from funscript_editor.utils.config import UI_CONFIG, VERSION
from funscript_editor.definitions import APP_DOCUMENTATION_DIR, CODE_DOCUMENTATION_DIR

class FunscriptEditorWindow(QtWidgets.QMainWindow):
    """ Funscript Editor window """

    def __init__(self):
        super(FunscriptEditorWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.__setup_video_player()
        self.__setup_funscript_visualizer()
        self.__setup_layout()
        self.__setup_ui_binding()
        self.__setup_modul_bindings()
        self.__setup_shortcut_bindings()
        self.__setup_variables()
        self.__setup_autosave_timer()
        self.setMouseTracking(True)

    __generateFunscript = QtCore.pyqtSignal()
    __logger = logging.getLogger(__name__)


    def mouseMoveEvent(self, event):
        """ Track the mouse in Qt Window """
        # On Windows the embedded mpv player do not get the mouse events
        if False:
            if platform.system() == 'Windows':
                self.video_player.send_mouse_pos(
                        event.x() - self.ui.videoPane.x(),
                        event.y() - self.ui.videoPane.y() - self.ui.menubar.height())


    def closeEvent(self, event):
        """ Implement for the Qt closeEvent handler """
        self.__save_funscript()

    def keyPressEvent(self, event):
        """ Implement for the Qt keyPressEvent handler """
        super(FunscriptEditorWindow, self).keyPressEvent(event)
        if event.key() == QtCore.Qt.Key_Delete: self.__delete_current_action()
        if event.key() == QtCore.Qt.Key_Space: self.video_player.toggle_play_pause()
        if event.key() == QtCore.Qt.Key_P: self.video_player.toggle_play_pause()
        if event.key() == QtCore.Qt.Key_Comma: self.video_player.show_prev_frame()
        if event.key() == QtCore.Qt.Key_Period: self.video_player.show_next_frame()
        if event.key() == QtCore.Qt.Key_BracketLeft: self.video_player.decrease_speed()
        if event.key() == QtCore.Qt.Key_BracketRight: self.video_player.increase_speed()
        if event.key() == QtCore.Qt.Key_W: self.video_player.move_stroke_indicator_relative(0,-1)
        if event.key() == QtCore.Qt.Key_S: self.video_player.move_stroke_indicator_relative(0,+1)
        if event.key() == QtCore.Qt.Key_A: self.video_player.move_stroke_indicator_relative(-1,0)
        if event.key() == QtCore.Qt.Key_D: self.video_player.move_stroke_indicator_relative(+1,0)
        if event.key() == QtCore.Qt.Key_End: self.video_player.seek_time(self.video_player.get_duration)
        if event.key() == QtCore.Qt.Key_Home: self.video_player.seek_time(0)

    def __setup_variables(self):
        self.funscript = None

    def __setup_autosave_timer(self):
        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.setInterval(UI_CONFIG['autosave_in_sec'] * 1000)
        self.autosave_timer.timeout.connect(self.__save_funscript)
        self.autosave_timer.start()

    def __setup_video_player(self):
        self.video_player = VideoPlayer(self.ui.videoPane, key_callback=self.__video_player_on_key_press)
        self.video_player.start()

    def __setup_funscript_visualizer(self):
        self.funscript_visualizer = FunscriptVisualizer(self.ui.animationPane)
        self.funscript_visualizer.start()

    def __setup_layout(self):
        self.ui.splitterHorizontal.setStretchFactor(0, UI_CONFIG['horizontal_stretch_factor'][0])
        self.ui.splitterHorizontal.setStretchFactor(1, UI_CONFIG['horizontal_stretch_factor'][1])
        self.ui.splitterVertical.setStretchFactor(0, UI_CONFIG['vertical_stretch_factor'][0])
        self.ui.splitterVertical.setStretchFactor(1, UI_CONFIG['vertical_stretch_factor'][1])

    def __setup_modul_bindings(self):
        self.funscript_visualizer.set_select_point_callback(self.video_player.seek_frame)
        self.video_player.set_frame_changed_callback(self.__frame_changed_handler)

    def __frame_changed_handler(self, frame_num):
        if not self.ui.seekBar.isSliderDown():
            self.ui.seekBar.setValue(frame_num)
        self.funscript_visualizer.set_frame(frame_num)
        if self.funscript is None: return
        self.ui.currentStrokeLabel.setText('{} ms'.format(\
                self.funscript.get_stroke_time(self.video_player.get_current_timestamp_in_millis)))
        self.ui.fastestStrokeLabel.setText('{} ms'.format(self.funscript.get_fastest_stroke()))
        self.ui.slowstStrokeLabel.setText('{} ms'.format(self.funscript.get_slowest_stroke()))
        self.ui.medianStrokesLabel.setText('{} ms'.format(self.funscript.get_median_stroke()))

    def __setup_ui_binding(self):
        self.ui.menuFile.addAction('Open (Ctrl+O)', self.__open_video)
        self.ui.menuFile.addAction('New (Ctrl+N)', self.__new_funscript)
        self.ui.menuFile.addAction('Save (Ctrl+S)', self.__save_funscript)
        self.ui.menuFile.addAction('Clear History (Ctrl+C)', self.__clear_funscript_history)

        helpMenu = self.ui.menubar.addMenu("Help")
        # TODO we schold use an http server to show the documentation
        helpMenu.addAction("App Documentation", lambda : webbrowser.open(os.path.join(APP_DOCUMENTATION_DIR, 'index.html')))
        helpMenu.addAction("Code Documentation", lambda : webbrowser.open(os.path.join(CODE_DOCUMENTATION_DIR, 'index.html')))
        helpMenu.addAction(str('Version '+VERSION))
        self.ui.seekBar.sliderReleased.connect(lambda: self.video_player.seek_frame(self.ui.seekBar.value()))

        self.__generateFunscript.connect(self.__generate_funscript)

    def __setup_shortcut_bindings(self):
        QtWidgets.QShortcut('CTRL+Left', self).activated.connect(self.__seek_prev)
        QtWidgets.QShortcut('CTRL+Right', self).activated.connect(self.__seek_next)
        QtWidgets.QShortcut('CTRL+Shift+Right', self).activated.connect(self.__seek_to_next_action)
        QtWidgets.QShortcut('CTRL+Shift+Left', self).activated.connect(self.__seek_to_prev_action)

        QtWidgets.QShortcut('CTRL++', self).activated.connect(self.video_player.inc_stroke_indicator)
        QtWidgets.QShortcut('CTRL+-', self).activated.connect(self.video_player.dec_stroke_indicator)

        QtWidgets.QShortcut('Shift+End', self).activated.connect(self.__seek_to_last_action)
        QtWidgets.QShortcut('Shift+Home', self).activated.connect(self.__seek_to_first_action)

        QtWidgets.QShortcut('0', self).activated.connect(lambda: self.__add_action(0))
        QtWidgets.QShortcut('1', self).activated.connect(lambda: self.__add_action(10))
        QtWidgets.QShortcut('2', self).activated.connect(lambda: self.__add_action(20))
        QtWidgets.QShortcut('3', self).activated.connect(lambda: self.__add_action(30))
        QtWidgets.QShortcut('4', self).activated.connect(lambda: self.__add_action(40))
        QtWidgets.QShortcut('5', self).activated.connect(lambda: self.__add_action(50))
        QtWidgets.QShortcut('6', self).activated.connect(lambda: self.__add_action(60))
        QtWidgets.QShortcut('7', self).activated.connect(lambda: self.__add_action(70))
        QtWidgets.QShortcut('8', self).activated.connect(lambda: self.__add_action(80))
        QtWidgets.QShortcut('9', self).activated.connect(lambda: self.__add_action(90))

        QtWidgets.QShortcut('CTRL+o', self).activated.connect(self.__open_video)
        QtWidgets.QShortcut('CTRL+c', self).activated.connect(self.__clear_funscript_history)
        QtWidgets.QShortcut('CTRL+g', self).activated.connect(lambda: self.__generateFunscript.emit())
        QtWidgets.QShortcut('CTRL+i', self).activated.connect(self.video_player.toogle_stroke_indicator_inversion)
        QtWidgets.QShortcut('CTRL+s', self).activated.connect(self.__save_funscript)
        QtWidgets.QShortcut('CTRL+Shift+Delete', self).activated.connect(self.__delete_folowing)
        QtWidgets.QShortcut('CTRL+n', self).activated.connect(self.__new_funscript)

    def __delete_folowing(self):
        reply = QtWidgets.QMessageBox.question(None, 'Delete Folowing Actions', 'Delete all folowing actions? ',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.statusBar().showMessage("Delete folowing actions ({})".format(datetime.now().strftime("%H:%M:%S")))
            if self.funscript is None: return
            self.funscript.delete_folowing_actions(self.video_player.get_current_timestamp_in_millis)
            self.funscript_visualizer.update()

    def __clear_funscript_history(self):
        if self.funscript is None: return
        if self.video_player is None: return
        if self.video_player.get_video_file is None: return
        self.__save_funscript()
        funscript_path = ''.join(self.video_player.get_video_file[:-4]) + '.funscript'
        num = 0
        while (os.path.exists(funscript_path + str(num))):
            os.remove(funscript_path + str(num))
            num += 1
        self.video_player.show_message("The funscript history was cleaned")

    def __seek_to_first_action(self):
        if self.funscript is not None:
            self.video_player.seek_time(self.funscript.get_first_action_time() / float(1000))

    def __seek_to_last_action(self):
        if self.funscript is not None:
            self.video_player.seek_time(self.funscript.get_last_action_time() / float(1000))

    def __seek_to_prev_action(self):
        if self.funscript is not None:
            self.video_player.seek_time( self.funscript.get_prev_action(
                self.video_player.get_current_timestamp_in_millis)['at'] / float(1000))

    def __seek_to_next_action(self):
        if self.funscript is not None:
            self.video_player.seek_time( self.funscript.get_next_action(
                self.video_player.get_current_timestamp_in_millis)['at'] / float(1000))

    def __seek_next(self):
        self.video_player.seek_time(\
                self.video_player.get_current_timestamp_in_millis / float(1000) +  UI_CONFIG['seek_next_sec'])

    def __seek_prev(self):
        self.video_player.seek_time(\
                max((0.0, self.video_player.get_current_timestamp_in_millis \
                / float(1000) - UI_CONFIG['seek_prev_sec'])))


    def __invert_actions(self):
        if self.funscript is None: return
        self.funscript.invert_actions()
        self.funscript_visualizer.update()

    def __video_player_on_key_press(self, key):
        if key == 'ctrl+g': self.__generateFunscript.emit()
        if key == 'ctrl+s': self.__save_funscript()
        if key == 'ctrl+right': self.__seek_next()
        if key == 'ctrl+left': self.__seek_prev()
        if key == 'ctrl+shift+right': self.__seek_to_next_action()
        if key == 'ctrl+shift+left': self.__seek_to_prev_action()
        if key == '0': self.__add_action(0)
        if key == '1': self.__add_action(10)
        if key == '2': self.__add_action(20)
        if key == '3': self.__add_action(30)
        if key == '4': self.__add_action(40)
        if key == '5': self.__add_action(50)
        if key == '6': self.__add_action(60)
        if key == '7': self.__add_action(70)
        if key == '8': self.__add_action(80)
        if key == '9': self.__add_action(90)
        if key == 'delete': self.__delete_current_action()
        if key == 'shift+end': self.__seek_to_last_action()
        if key == 'shift+home': self.__seek_to_first_action()
        if key == 'ctrl+n': self.__new_funscript()
        if key == 'ctrl+o': self.__open_video()
        if key == 'ctrl+c': self.__clear_funscript_history()

    def __show_message(self, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message+' ')
        msg.setWindowTitle("Funscript Editor Info")
        msg.exec_()

    def __save_funscript(self):
        if self.funscript is None: return
        if not self.funscript.changed: return
        self.funscript.save(''.join(self.video_player.get_video_file[:-4]) + '.funscript')
        self.statusBar().showMessage("Funscript Saved ({})".format(datetime.now().strftime("%H:%M:%S")))

    def __add_action(self, value):
        if self.funscript is None: return
        self.funscript.add_action(value, self.video_player.get_current_timestamp_in_millis)
        self.funscript_visualizer.update()

    def __open_video(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(
                None,
                caption="Select Video File",
                directory=QtCore.QDir.currentPath())
        if len(file_name) < 1: return
        if not any(file_name[0].lower().endswith(x) for x in ['.mkv', '.mp4']): return

        self.video_player.open(file_name[0])

        funscript_file = ''.join(file_name[0][:-4]) + '.funscript'
        if os.path.exists(funscript_file):
            self.funscript, file_path = Funscript.load(file_name[0], funscript_file)
            self.video_player.show_message('load: ' + str(file_path.split('.')[-1]))
        else:
            self.funscript = Funscript(fps=self.video_player.get_fps)

        self.video_player.set_funscript(self.funscript)
        self.funscript_visualizer.set_funscript(self.funscript)
        self.ui.seekBar.setMaximum(max((0, self.video_player.get_length-1)))
        self.ui.seekBar.setValue(0)

    def __new_funscript(self):
        self.funscript = Funscript(fps=self.video_player.get_fps)
        self.video_player.set_funscript(self.funscript)
        self.funscript_visualizer.set_funscript(self.funscript)

    def __generator_status_changed(self, current):
        self.statusBar().showMessage("{} ({})".format(current, datetime.now().strftime("%H:%M:%S")))

    def __generate_funscript(self):
        if self.funscript is None: return
        if self.video_player is None: return
        if self.video_player.get_video_file is None: return
        start_frame = self.video_player.get_current_frame
        next_action = self.funscript.get_next_action(self.video_player.get_current_timestamp_in_millis+100)
        if next_action['at'] > self.video_player.get_current_timestamp_in_millis+100:
            end_frame = self.video_player.millisec_to_frame(next_action['at'])
        else:
            end_frame = -1

        self.__logger.info("Stop at Frame {}".format(end_frame))

        reply = QtWidgets.QMessageBox.question(None, 'Track Men', 'Do you want to track the Men? ',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        trackMen = True if reply == QtWidgets.QMessageBox.Yes else False

        self.video_player.set_indicate_bussy(True)
        self.funscript_generator = FunscriptGenerator(
                FunscriptGeneratorParameter(
                    video_path = self.video_player.get_video_file,
                    start_frame = start_frame,
                    end_frame = end_frame,
                    track_men = trackMen
                ),
                self.funscript)
        self.funscript_generator.funscriptCompleted.connect(self.__funscript_generated)
        self.funscript_generator.processStatus.connect(self.__generator_status_changed)
        self.funscript_generator.start()

    def __funscript_generated(self, funscript, status, success):
        self.video_player.set_funscript(self.funscript)
        self.funscript_visualizer.set_funscript(self.funscript)
        self.video_player.set_indicate_bussy(False)
        self.__save_funscript()
        self.video_player.show_message(status)
        if success:
            self.__logger.info(status)
        else:
            self.__logger.error(status)

    def __delete_current_action(self):
        if self.funscript is None: return
        self.funscript.delete_action(self.video_player.get_current_timestamp_in_millis)
        self.funscript_visualizer.update()

