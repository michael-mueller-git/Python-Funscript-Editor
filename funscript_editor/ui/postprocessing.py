from PyQt5 import QtCore, QtGui, QtWidgets
from simplification.cutil import (
    simplify_coords_idx,
    simplify_coords_vw_idx,
)

import numpy as np
import pyqtgraph as pg
from funscript_editor.ui.cut_tracking_result import Slider

class PostprocessingWidget(QtWidgets.QWidget):
    def __init__(self, metric, raw_score, parent=None):
        super(QtWidgets.QWidget, self).__init__(parent=parent)
        pg.setConfigOption("background","w")
        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self.metric = metric
        self.raw_score_idx = [x for x in range(len(raw_score))]
        self.raw_score = raw_score
        self.raw_score_np = [[i, raw_score[i]] for i in range(len(raw_score))]

        self.verticalLayout.addWidget(QtWidgets.QLabel("Postprocessing"))

        self.w1 = Slider("Epsilon", 100, 13)
        self.verticalLayout.addWidget(self.w1)

        self.btn = QtWidgets.QPushButton("OK")
        self.verticalLayout.addWidget(self.btn)
        self.btn.clicked.connect(self.confirm)

        self.win = pg.GraphicsLayoutWidget(title="Postprocessing")
        self.verticalLayout.addWidget(self.win)
        self.p6 = self.win.addPlot(title="Funscript Action")
        self.p6.setYRange(0,100)
        self.p6.setXRange(0,len(raw_score))
        self.p6.vb.setLimits(xMin=0, xMax=len(raw_score), yMin=0, yMax=100)
        self.p6.setMouseEnabled(y=False)
        self.curve_raw = self.p6.plot(pen=pg.mkPen("b", width=2.0))
        self.curve_result = self.p6.plot(pen=pg.mkPen("r", width=2.0))

        self.w1.slider.valueChanged.connect(self.update_plot)
        self.result = []
        self.update_plot()

    postprocessingCompleted = QtCore.pyqtSignal(str, list)


    def confirm(self):
        self.hide()
        self.postprocessingCompleted.emit(self.metric, self.result)
        self.close()


    def update_plot(self):
        self.curve_raw.setData(self.raw_score_idx, self.raw_score)
        self.result = simplify_coords_idx(self.raw_score_np, float(self.w1.x) / 10.0)
        self.curve_result.setData(self.result, [val for idx,val in enumerate(self.raw_score) if idx in self.result])
