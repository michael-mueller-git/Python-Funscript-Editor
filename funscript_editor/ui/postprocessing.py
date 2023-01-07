from PyQt5 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
from funscript_editor.ui.cut_tracking_result import Slider

class PostprocessingWidget(QtWidgets.QWidget):
    def __init__(self, metric, raw_score, parent=None):
        super(QtWidgets.QWidget, self).__init__(parent=parent)
        pg.setConfigOption("background","w")
        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        self.metric = metric
        self.raw_score = raw_score

        self.verticalLayout.addWidget(QtWidgets.QLabel("Postprocessing"))

        self.w1 = Slider("Epsilon", 100, 10)
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
        self.update_plot()

        self.w1.slider.valueChanged.connect(self.update_plot)

        self.result = { }

    postprocessingCompleted = QtCore.pyqtSignal(str, dict)


    def confirm(self):
        self.hide()
        self.postprocessingCompleted.emit(self.metric, self.result)
        self.close()


    def update_plot(self):
        self.curve_raw.setData(self.raw_score)
