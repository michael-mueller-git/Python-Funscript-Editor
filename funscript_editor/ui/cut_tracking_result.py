import sys
import random

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from PyQt5 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
from funscript_editor.algorithms.signal import Signal


class Slider(QWidget):
    def __init__(self, prefix_txt, maximum, init_value=0):
        super(Slider, self).__init__(parent=None)
        self.horizontalLayout = QHBoxLayout(self)
        self.prefix_txt = prefix_txt
        self.label = QLabel(self)
        self.horizontalLayout.addWidget(self.label)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.horizontalLayout.addWidget(self.slider)
        self.resize(self.sizeHint())

        self.maximum = maximum
        self.slider.setMaximum(maximum)
        self.slider.setSingleStep(1)

        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = init_value
        self.setLabelValue(self.slider.value())
        self.slider.setValue(init_value)

    def setLabelValue(self, value):
        self.x = value
        self.label.setText(f"{self.prefix_txt} {self.x: <8}")


class CutTrackingResultWidget(QtWidgets.QDialog):
    def __init__(self, raw_score, metrics, parent=None):
        super(QWidget, self).__init__(parent=parent)
        pg.setConfigOption("background","w")
        self.verticalLayout = QVBoxLayout(self)

        self.raw_score = raw_score
        data_len = max([len(raw_score[k]) for k in raw_score if k in metrics])

        self.verticalLayout.addWidget(QLabel("Cut tracking result before postprocessing"))

        self.w1 = Slider("Start at Frame", data_len, 0)
        self.verticalLayout.addWidget(self.w1)

        self.w2 = Slider("Stop at Frame", data_len, data_len)
        self.verticalLayout.addWidget(self.w2)

        self.btn = QPushButton("OK")
        self.verticalLayout.addWidget(self.btn)
        self.btn.clicked.connect(self.confirm)

        self.win = pg.GraphicsLayoutWidget(title="Cut Tracking result")
        self.verticalLayout.addWidget(self.win)
        self.p6 = self.win.addPlot(title="Tracking result")
        self.p6.setYRange(0,100)
        self.p6.setXRange(0,data_len)
        self.p6.vb.setLimits(xMin=0, xMax=data_len, yMin=0, yMax=100)
        self.p6.setMouseEnabled(y=False)
        colors = ["b", "r", "g", "c", "m", "k", "y"]
        self.curve = { k: self.p6.plot(pen=pg.mkPen(c, width=2.0)) for c,k in zip(colors, metrics) }

        self.w1.slider.valueChanged.connect(self.update_plot)
        self.w2.slider.valueChanged.connect(self.update_plot)

        self.result = {
            "start": 0,
            "stop": data_len
        }
        self.update_plot()
        self.close_with_ok = False

    cutCompleted = QtCore.pyqtSignal(dict)


    def closeEvent(self, event):
        event.accept()
        if not self.close_with_ok:
            sys.exit(1)


    def confirm(self):
        self.close_with_ok = True
        self.hide()
        self.cutCompleted.emit(self.result)
        self.close()


    def update_plot(self):
        start = self.w1.x
        stop = self.w2.x
        self.result = {
            "start": start,
            "stop": stop
        }

        for k in self.curve:
            if start < stop:
                new_score = Signal.scale(self.raw_score[k][start:stop], 0 , 100)
                self.curve[k].setData([start+i for i in range(stop-start)], new_score)
            else:
                self.curve[k].setData([])

