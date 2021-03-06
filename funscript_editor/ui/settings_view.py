# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings_view.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(550, 500)
        self.gridLayout_4 = QtWidgets.QGridLayout(Form)
        self.gridLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_4.setContentsMargins(-1, 6, 6, 6)
        self.gridLayout_4.setSpacing(6)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 0, 0, 1, 1)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_4 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.videoTypeComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.videoTypeComboBox.sizePolicy().hasHeightForWidth())
        self.videoTypeComboBox.setSizePolicy(sizePolicy)
        self.videoTypeComboBox.setObjectName("videoTypeComboBox")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.videoTypeComboBox)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.trackingMetricComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trackingMetricComboBox.sizePolicy().hasHeightForWidth())
        self.trackingMetricComboBox.setSizePolicy(sizePolicy)
        self.trackingMetricComboBox.setObjectName("trackingMetricComboBox")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.trackingMetricComboBox)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.trackingMethodComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trackingMethodComboBox.sizePolicy().hasHeightForWidth())
        self.trackingMethodComboBox.setSizePolicy(sizePolicy)
        self.trackingMethodComboBox.setObjectName("trackingMethodComboBox")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.trackingMethodComboBox)
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.processingSpeedComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.processingSpeedComboBox.sizePolicy().hasHeightForWidth())
        self.processingSpeedComboBox.setSizePolicy(sizePolicy)
        self.processingSpeedComboBox.setObjectName("processingSpeedComboBox")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.processingSpeedComboBox)
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.numberOfTrackerComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.numberOfTrackerComboBox.sizePolicy().hasHeightForWidth())
        self.numberOfTrackerComboBox.setSizePolicy(sizePolicy)
        self.numberOfTrackerComboBox.setObjectName("numberOfTrackerComboBox")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.numberOfTrackerComboBox)
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.pointsComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pointsComboBox.sizePolicy().hasHeightForWidth())
        self.pointsComboBox.setSizePolicy(sizePolicy)
        self.pointsComboBox.setObjectName("pointsComboBox")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.pointsComboBox)
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.additionalPointsComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.additionalPointsComboBox.sizePolicy().hasHeightForWidth())
        self.additionalPointsComboBox.setSizePolicy(sizePolicy)
        self.additionalPointsComboBox.setObjectName("additionalPointsComboBox")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.additionalPointsComboBox)
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.horizontalLayoutTopPointOffset = QtWidgets.QHBoxLayout()
        self.horizontalLayoutTopPointOffset.setObjectName("horizontalLayoutTopPointOffset")
        self.topPointOffsetSpinBox = QtWidgets.QSpinBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.topPointOffsetSpinBox.sizePolicy().hasHeightForWidth())
        self.topPointOffsetSpinBox.setSizePolicy(sizePolicy)
        self.topPointOffsetSpinBox.setPrefix("")
        self.topPointOffsetSpinBox.setMinimum(-100)
        self.topPointOffsetSpinBox.setMaximum(100)
        self.topPointOffsetSpinBox.setSingleStep(5)
        self.topPointOffsetSpinBox.setProperty("value", 10)
        self.topPointOffsetSpinBox.setObjectName("topPointOffsetSpinBox")
        self.horizontalLayoutTopPointOffset.addWidget(self.topPointOffsetSpinBox)
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setObjectName("label_11")
        self.horizontalLayoutTopPointOffset.addWidget(self.label_11)
        self.formLayout.setLayout(8, QtWidgets.QFormLayout.FieldRole, self.horizontalLayoutTopPointOffset)
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.horizontalLayoutBottomPointOffset = QtWidgets.QHBoxLayout()
        self.horizontalLayoutBottomPointOffset.setObjectName("horizontalLayoutBottomPointOffset")
        self.bottomPointOffsetSpinBox = QtWidgets.QSpinBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bottomPointOffsetSpinBox.sizePolicy().hasHeightForWidth())
        self.bottomPointOffsetSpinBox.setSizePolicy(sizePolicy)
        self.bottomPointOffsetSpinBox.setMinimum(-100)
        self.bottomPointOffsetSpinBox.setMaximum(100)
        self.bottomPointOffsetSpinBox.setSingleStep(5)
        self.bottomPointOffsetSpinBox.setProperty("value", -10)
        self.bottomPointOffsetSpinBox.setObjectName("bottomPointOffsetSpinBox")
        self.horizontalLayoutBottomPointOffset.addWidget(self.bottomPointOffsetSpinBox)
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setObjectName("label_12")
        self.horizontalLayoutBottomPointOffset.addWidget(self.label_12)
        self.formLayout.setLayout(9, QtWidgets.QFormLayout.FieldRole, self.horizontalLayoutBottomPointOffset)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.docsButton = QtWidgets.QPushButton(Form)
        self.docsButton.setObjectName("docsButton")
        self.horizontalLayout.addWidget(self.docsButton)
        self.okButton = QtWidgets.QPushButton(Form)
        self.okButton.setObjectName("okButton")
        self.horizontalLayout.addWidget(self.okButton)
        self.formLayout.setLayout(10, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout)
        self.gridLayout_4.addLayout(self.formLayout, 1, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem2, 2, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_4.setText(_translate("Form", " MTFG Settings"))
        self.label.setText(_translate("Form", "Video Type:"))
        self.label_2.setText(_translate("Form", "Tracking Metric:"))
        self.label_3.setText(_translate("Form", "Tracking Method:"))
        self.label_10.setText(_translate("Form", "Processing Speed:"))
        self.label_5.setText(_translate("Form", "Number of Tracker:"))
        self.label_6.setText(_translate("Form", "Points:"))
        self.label_7.setText(_translate("Form", "Additional Points:"))
        self.label_8.setText(_translate("Form", "Top Points Offset:"))
        self.label_11.setText(_translate("Form", " Recommended = +10 "))
        self.label_9.setText(_translate("Form", "Bottom Points Offset:"))
        self.label_12.setText(_translate("Form", " Recommended = -10 "))
        self.docsButton.setText(_translate("Form", "Documentation"))
        self.okButton.setText(_translate("Form", "OK"))
