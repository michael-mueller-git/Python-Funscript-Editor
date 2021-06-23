# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'funscript_editor_view.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(959, 834)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.mainLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.mainLayout.setObjectName("mainLayout")
        self.splitterVertical = QtWidgets.QSplitter(self.centralwidget)
        self.splitterVertical.setOrientation(QtCore.Qt.Vertical)
        self.splitterVertical.setObjectName("splitterVertical")
        self.splitterHorizontal = QtWidgets.QSplitter(self.splitterVertical)
        self.splitterHorizontal.setOrientation(QtCore.Qt.Horizontal)
        self.splitterHorizontal.setObjectName("splitterHorizontal")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.splitterHorizontal)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.videoLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.videoLayout.setContentsMargins(0, 0, 0, 0)
        self.videoLayout.setObjectName("videoLayout")
        self.videoPane = QtWidgets.QWidget(self.verticalLayoutWidget)
        self.videoPane.setObjectName("videoPane")
        self.videoLayout.addWidget(self.videoPane)
        spacerItem = QtWidgets.QSpacerItem(40, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.videoLayout.addItem(spacerItem)
        self.seekBar = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.seekBar.setOrientation(QtCore.Qt.Horizontal)
        self.seekBar.setObjectName("seekBar")
        self.videoLayout.addWidget(self.seekBar)
        self.settingsLayout = QtWidgets.QGroupBox(self.splitterHorizontal)
        self.settingsLayout.setTitle("")
        self.settingsLayout.setObjectName("settingsLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.settingsLayout)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.shortcutsGroupBox = QtWidgets.QGroupBox(self.settingsLayout)
        self.shortcutsGroupBox.setObjectName("shortcutsGroupBox")
        self.verticalLayout_3 = QtWidgets.QFormLayout(self.shortcutsGroupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label.setObjectName("label")
        self.verticalLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.label_2)
        self.label_3 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_4)
        self.label_5 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.label_6 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.label_6)
        self.label_7 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.label_8 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.label_8)
        self.label_9 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.label_10 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.label_10)
        self.label_12 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_3.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.label_14 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_3.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.label_14)
        self.label_16 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_16.setObjectName("label_16")
        self.verticalLayout_3.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.label_18 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_18.setObjectName("label_18")
        self.verticalLayout_3.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.label_18)
        self.label_19 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_19.setObjectName("label_19")
        self.verticalLayout_3.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.label_20 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_20.setObjectName("label_20")
        self.verticalLayout_3.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.label_20)
        self.label_21 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_21.setObjectName("label_21")
        self.verticalLayout_3.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.label_22 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_22.setObjectName("label_22")
        self.verticalLayout_3.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.label_22)
        self.label_23 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_23.setObjectName("label_23")
        self.verticalLayout_3.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_23)
        self.label_26 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_26.setObjectName("label_26")
        self.verticalLayout_3.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.label_26)
        self.label_24 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_24.setObjectName("label_24")
        self.verticalLayout_3.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.label_27 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_27.setObjectName("label_27")
        self.verticalLayout_3.setWidget(10, QtWidgets.QFormLayout.FieldRole, self.label_27)
        self.label_25 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_25.setObjectName("label_25")
        self.verticalLayout_3.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_25)
        self.label_28 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_28.setObjectName("label_28")
        self.verticalLayout_3.setWidget(11, QtWidgets.QFormLayout.FieldRole, self.label_28)
        self.label_29 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_29.setObjectName("label_29")
        self.verticalLayout_3.setWidget(13, QtWidgets.QFormLayout.LabelRole, self.label_29)
        self.label_31 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_31.setObjectName("label_31")
        self.verticalLayout_3.setWidget(13, QtWidgets.QFormLayout.FieldRole, self.label_31)
        self.label_30 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_30.setObjectName("label_30")
        self.verticalLayout_3.setWidget(14, QtWidgets.QFormLayout.LabelRole, self.label_30)
        self.label_32 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_32.setObjectName("label_32")
        self.verticalLayout_3.setWidget(14, QtWidgets.QFormLayout.FieldRole, self.label_32)
        self.label_34 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_34.setObjectName("label_34")
        self.verticalLayout_3.setWidget(15, QtWidgets.QFormLayout.LabelRole, self.label_34)
        self.label_36 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_36.setObjectName("label_36")
        self.verticalLayout_3.setWidget(15, QtWidgets.QFormLayout.FieldRole, self.label_36)
        self.label_35 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_35.setObjectName("label_35")
        self.verticalLayout_3.setWidget(12, QtWidgets.QFormLayout.FieldRole, self.label_35)
        self.label_33 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_33.setObjectName("label_33")
        self.verticalLayout_3.setWidget(12, QtWidgets.QFormLayout.LabelRole, self.label_33)
        self.label_37 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_37.setObjectName("label_37")
        self.verticalLayout_3.setWidget(16, QtWidgets.QFormLayout.LabelRole, self.label_37)
        self.label_38 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_38.setObjectName("label_38")
        self.verticalLayout_3.setWidget(16, QtWidgets.QFormLayout.FieldRole, self.label_38)
        self.label_39 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_39.setObjectName("label_39")
        self.verticalLayout_3.setWidget(17, QtWidgets.QFormLayout.LabelRole, self.label_39)
        self.label_40 = QtWidgets.QLabel(self.shortcutsGroupBox)
        self.label_40.setObjectName("label_40")
        self.verticalLayout_3.setWidget(17, QtWidgets.QFormLayout.FieldRole, self.label_40)
        self.verticalLayout_2.addWidget(self.shortcutsGroupBox)
        self.strokeGroupBox = QtWidgets.QGroupBox(self.settingsLayout)
        self.strokeGroupBox.setObjectName("strokeGroupBox")
        self.verticalLayout_4 = QtWidgets.QFormLayout(self.strokeGroupBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_11 = QtWidgets.QLabel(self.strokeGroupBox)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.fastestStrokeLabel = QtWidgets.QLabel(self.strokeGroupBox)
        self.fastestStrokeLabel.setObjectName("fastestStrokeLabel")
        self.verticalLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.fastestStrokeLabel)
        self.label_13 = QtWidgets.QLabel(self.strokeGroupBox)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.medianStrokesLabel = QtWidgets.QLabel(self.strokeGroupBox)
        self.medianStrokesLabel.setObjectName("medianStrokesLabel")
        self.verticalLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.medianStrokesLabel)
        self.label_15 = QtWidgets.QLabel(self.strokeGroupBox)
        self.label_15.setObjectName("label_15")
        self.verticalLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.slowstStrokeLabel = QtWidgets.QLabel(self.strokeGroupBox)
        self.slowstStrokeLabel.setObjectName("slowstStrokeLabel")
        self.verticalLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.slowstStrokeLabel)
        self.label_17 = QtWidgets.QLabel(self.strokeGroupBox)
        self.label_17.setObjectName("label_17")
        self.verticalLayout_4.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.currentStrokeLabel = QtWidgets.QLabel(self.strokeGroupBox)
        self.currentStrokeLabel.setObjectName("currentStrokeLabel")
        self.verticalLayout_4.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.currentStrokeLabel)
        self.verticalLayout_2.addWidget(self.strokeGroupBox)
        self.animationPane = QtWidgets.QLabel(self.splitterVertical)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.animationPane.sizePolicy().hasHeightForWidth())
        self.animationPane.setSizePolicy(sizePolicy)
        self.animationPane.setText("")
        self.animationPane.setObjectName("animationPane")
        self.mainLayout.addWidget(self.splitterVertical)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 959, 34))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Video Player with Signal"))
        self.shortcutsGroupBox.setTitle(_translate("MainWindow", "Keyboard Shortcuts"))
        self.label.setText(_translate("MainWindow", "P, Space"))
        self.label_2.setText(_translate("MainWindow", "Pause / Playback"))
        self.label_3.setText(_translate("MainWindow", "."))
        self.label_4.setText(_translate("MainWindow", "Next frame"))
        self.label_5.setText(_translate("MainWindow", ","))
        self.label_6.setText(_translate("MainWindow", "Previus frame"))
        self.label_7.setText(_translate("MainWindow", "["))
        self.label_8.setText(_translate("MainWindow", "Decrease speed"))
        self.label_9.setText(_translate("MainWindow", "]"))
        self.label_10.setText(_translate("MainWindow", "Increase speed"))
        self.label_12.setText(_translate("MainWindow", "Ctrl+Shift+Left"))
        self.label_14.setText(_translate("MainWindow", "Previous Action Point"))
        self.label_16.setText(_translate("MainWindow", "Ctrl+Shift+Right"))
        self.label_18.setText(_translate("MainWindow", "Next Action Point"))
        self.label_19.setText(_translate("MainWindow", "Shift+L"))
        self.label_20.setText(_translate("MainWindow", "Loop Video"))
        self.label_21.setText(_translate("MainWindow", "W"))
        self.label_22.setText(_translate("MainWindow", "Move Stroke Indicator Up"))
        self.label_23.setText(_translate("MainWindow", "A"))
        self.label_26.setText(_translate("MainWindow", "Move Stroke Indicator Left"))
        self.label_24.setText(_translate("MainWindow", "S"))
        self.label_27.setText(_translate("MainWindow", "Move Stroke Indicator Down"))
        self.label_25.setText(_translate("MainWindow", "D"))
        self.label_28.setText(_translate("MainWindow", "Move Stroke Indicator Right"))
        self.label_29.setText(_translate("MainWindow", "Ctrl+Plus"))
        self.label_31.setText(_translate("MainWindow", "Increase Stoke Indicator"))
        self.label_30.setText(_translate("MainWindow", "Ctrl+Minus"))
        self.label_32.setText(_translate("MainWindow", "Decrease Stroke Indicator"))
        self.label_34.setText(_translate("MainWindow", "Ctrl+G"))
        self.label_36.setText(_translate("MainWindow", "Start Funscript Generator"))
        self.label_35.setText(_translate("MainWindow", "Invert Stroke indicator"))
        self.label_33.setText(_translate("MainWindow", "Ctrl+I"))
        self.label_37.setText(_translate("MainWindow", "Shift+Pos1"))
        self.label_38.setText(_translate("MainWindow", "First Action"))
        self.label_39.setText(_translate("MainWindow", "Shift+Ende"))
        self.label_40.setText(_translate("MainWindow", "Last Action"))
        self.strokeGroupBox.setTitle(_translate("MainWindow", "Statistics"))
        self.label_11.setText(_translate("MainWindow", "Fastest Stroke:"))
        self.fastestStrokeLabel.setText(_translate("MainWindow", "0 ms"))
        self.label_13.setText(_translate("MainWindow", "Median Strokes:"))
        self.medianStrokesLabel.setText(_translate("MainWindow", "0 ms"))
        self.label_15.setText(_translate("MainWindow", "Slowest Stroke:"))
        self.slowstStrokeLabel.setText(_translate("MainWindow", "0 ms"))
        self.label_17.setText(_translate("MainWindow", "Current Stroke"))
        self.currentStrokeLabel.setText(_translate("MainWindow", "0 ms"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
