from PyQt5 import QtCore, QtGui, QtWidgets
from simplification.cutil import (
    simplify_coords_idx,
    simplify_coords_vw_idx,
)

import numpy as np
import pyqtgraph as pg

from funscript_editor.algorithms.signal import Signal
from funscript_editor.ui.cut_tracking_result import Slider


class PostprocessingWidget(QtWidgets.QWidget):
    def __init__(self, metric, raw_score, video_info, parent=None):
        super(QtWidgets.QWidget, self).__init__(parent=parent)
        pg.setConfigOption("background","w")
        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self.video_info = video_info
        self.metric = metric
        self.raw_score_idx = [x for x in range(len(raw_score))]
        self.raw_score = raw_score
        self.raw_score_np = [[i, raw_score[i]] for i in range(len(raw_score))]

        self.verticalLayout.addWidget(QtWidgets.QLabel(f"Postprocessing for {metric}"))

        self.tabs = QtWidgets.QTabWidget()
        self.tabs_content = {}
        self.add_rdp_tab()
        self.add_vw_tab()
        self.add_custom_tab()
        self.verticalLayout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.update_plot)

        self.ok_btn = QtWidgets.QPushButton("OK")
        self.verticalLayout.addWidget(self.ok_btn)
        self.ok_btn.clicked.connect(self.confirm)

        self.win = pg.GraphicsLayoutWidget(title="Postprocessing")
        self.verticalLayout.addWidget(self.win)
        self.p6 = self.win.addPlot(title="Funscript Action")
        self.p6.setYRange(0, 100)
        self.p6.setXRange(0,len(raw_score))
        self.p6.vb.setLimits(xMin=0, xMax=len(raw_score), yMin=0, yMax=100)
        self.p6.setMouseEnabled(y=False)
        self.curve_raw = self.p6.plot(pen=pg.mkPen("b", width=2.0))
        self.curve_result = self.p6.plot(pen=pg.mkPen("r", width=2.0))
        self.curve_raw.setData(self.raw_score_idx, self.raw_score)

        self.result_idx = []
        self.result_val = []

        self.update_plot()


    postprocessingCompleted = QtCore.pyqtSignal(str, list, list)


    def add_rdp_tab(self):
        tab_name = "Ramer–Douglas–Peucker"
        self.tabs_content[tab_name] = {"main": QtWidgets.QWidget(), "widgets": {}}
        self.tabs_content[tab_name]["main"].layout = QtWidgets.QVBoxLayout(self)
        self.tabs_content[tab_name]["widgets"]["epsilon"] = Slider("Epsilon", 100, 10)
        self.tabs_content[tab_name]["main"].layout.addWidget(self.tabs_content[tab_name]["widgets"]["epsilon"])
        self.tabs_content[tab_name]["widgets"]["epsilon"].slider.valueChanged.connect(self.update_plot)
        self.tabs_content[tab_name]["main"].setLayout(self.tabs_content[tab_name]["main"].layout)
        self.tabs.addTab(self.tabs_content[tab_name]["main"], tab_name)


    def add_vw_tab(self):
        tab_name = "Visvalingam-Whyatt"
        self.tabs_content[tab_name] = {"main": QtWidgets.QWidget(), "widgets": {}}
        self.tabs_content[tab_name]["main"].layout = QtWidgets.QVBoxLayout(self)
        self.tabs_content[tab_name]["widgets"]["epsilon"] = Slider("Epsilon", 200, 50)
        self.tabs_content[tab_name]["main"].layout.addWidget(self.tabs_content[tab_name]["widgets"]["epsilon"])
        self.tabs_content[tab_name]["widgets"]["epsilon"].slider.valueChanged.connect(self.update_plot)
        self.tabs_content[tab_name]["main"].setLayout(self.tabs_content[tab_name]["main"].layout)
        self.tabs.addTab(self.tabs_content[tab_name]["main"], tab_name)


    def add_custom_tab(self):
        tab_name = "Custom"
        self.tabs_content[tab_name] = {"main": QtWidgets.QWidget(), "widgets": {}}
        self.tabs_content[tab_name]["main"].layout = QtWidgets.QVBoxLayout(self)

        self.tabs_content[tab_name]["widgets"]["points"] = QtWidgets.QComboBox()
        self.tabs_content[tab_name]["widgets"]["points"].addItems(["Local Min Max", "Direction Changed"])
        self.tabs_content[tab_name]["widgets"]["points"].currentIndexChanged.connect(self.update_plot)

        self.tabs_content[tab_name]["widgets"]["high_second_derivate"] = QtWidgets.QCheckBox("High Second Derivate")
        self.tabs_content[tab_name]["widgets"]["high_second_derivate"].stateChanged.connect(self.update_plot)

        self.tabs_content[tab_name]["widgets"]["distance_minimization"] = QtWidgets.QCheckBox("Distance Minimization")
        self.tabs_content[tab_name]["widgets"]["distance_minimization"].stateChanged.connect(self.update_plot)

        self.tabs_content[tab_name]["widgets"]["evenly_intermediate"] = QtWidgets.QCheckBox("Evenly Intermediate")
        self.tabs_content[tab_name]["widgets"]["evenly_intermediate"].stateChanged.connect(self.update_plot)

        self.tabs_content[tab_name]["widgets"]["runs"] = Slider("Additionl Points Algorthm Runs", 8, 2)
        self.tabs_content[tab_name]["widgets"]["runs"].slider.valueChanged.connect(self.update_plot)

        self.tabs_content[tab_name]["main"].layout.addWidget(self.tabs_content[tab_name]["widgets"]["points"])
        self.tabs_content[tab_name]["main"].layout.addWidget(self.tabs_content[tab_name]["widgets"]["high_second_derivate"])
        self.tabs_content[tab_name]["main"].layout.addWidget(self.tabs_content[tab_name]["widgets"]["distance_minimization"])
        self.tabs_content[tab_name]["main"].layout.addWidget(self.tabs_content[tab_name]["widgets"]["evenly_intermediate"])
        self.tabs_content[tab_name]["main"].layout.addWidget(self.tabs_content[tab_name]["widgets"]["runs"])

        self.tabs_content[tab_name]["main"].setLayout(self.tabs_content[tab_name]["main"].layout)
        self.tabs.addTab(self.tabs_content[tab_name]["main"], tab_name)


    def get_current_tab_name(self) -> str:
        return self.tabs.tabText(self.tabs.currentIndex())


    def confirm(self):
        self.hide()
        self.postprocessingCompleted.emit(self.metric, self.result_idx, self.result_val)
        self.close()


    def update_plot(self):
        current_tab_name = self.get_current_tab_name()

        if current_tab_name == "Ramer–Douglas–Peucker":
            self.result_idx = simplify_coords_idx(self.raw_score_np, float(self.tabs_content[current_tab_name]["widgets"]["epsilon"].x) / 10.0)
            self.result_val = [val for idx,val in enumerate(self.raw_score) if idx in self.result_idx]
            self.curve_result.setData(self.result_idx, self.result_val)
            return

        if current_tab_name == "Visvalingam-Whyatt":
            self.result_idx = simplify_coords_vw_idx(self.raw_score_np, float(self.tabs_content[current_tab_name]["widgets"]["epsilon"].x) / 1.0)
            self.result_val = [val for idx,val in enumerate(self.raw_score) if idx in self.result_idx]
            self.curve_result.setData(self.result_idx, self.result_val)
            return

        if current_tab_name == "Custom":
            base_algo = self.tabs_content[current_tab_name]["widgets"]["points"].currentText()
            runs = self.tabs_content[current_tab_name]["widgets"]["runs"].x

            base_point_algorithm = Signal.BasePointAlgorithm.local_min_max
            if base_algo == 'Direction Changed':
                base_point_algorithm = Signal.BasePointAlgorithm.direction_changes

            additional_points_algorithms = []
            if self.tabs_content[current_tab_name]["widgets"]["high_second_derivate"].isChecked():
                additional_points_algorithms.append(Signal.AdditionalPointAlgorithm.high_second_derivative)

            if self.tabs_content[current_tab_name]["widgets"]["distance_minimization"].isChecked():
                additional_points_algorithms.append(Signal.AdditionalPointAlgorithm.distance_minimization)

            if self.tabs_content[current_tab_name]["widgets"]["evenly_intermediate"].isChecked():
                additional_points_algorithms.append(Signal.AdditionalPointAlgorithm.evenly_intermediate)

            signal = Signal(self.video_info.fps)
            self.result_idx = signal.decimate(
                    self.raw_score,
                    base_point_algorithm,
                    additional_points_algorithms,
                    additional_points_repetitions = runs
            )
            self.result_val = [val for idx,val in enumerate(self.raw_score) if idx in self.result_idx]
            self.curve_result.setData(self.result_idx, self.result_val)
            return
