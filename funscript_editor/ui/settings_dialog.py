""" Settings Dialog for the Funscript Generator """
import json
import os

import funscript_editor.ui.settings_view as settings_view

from funscript_editor.utils.config import PROJECTION
from funscript_editor.definitions import CONFIG_DIR

from PyQt5 import QtWidgets, QtCore, QtGui

class SettingsDialog(QtWidgets.QDialog):
    """ Settings Dialog

    Args:
        settings (dict): dict where to store the settings
        include_vr (bool): include vr options
    """

    def __init__(self, settings: dict, include_vr: bool = True):
        super(SettingsDialog, self).__init__()
        self.include_vr = include_vr
        self.ui = settings_view.Ui_Form()
        self.form = QtWidgets.QDialog()
        self.ui.setupUi(self.form)
        self.form.setWindowTitle("MTFG Settings")
        self.settings = settings
        self.settings_file = os.path.join(CONFIG_DIR, "dialog_settings.json")
        self.__setup_ui_bindings()
        self.__setup_combo_boxes()
        self.__setup_dialog_elements()
        self.__load_settings()


    #: apply settings event
    applySettings = QtCore.pyqtSignal()


    def __setup_dialog_elements(self):
        self.dialog_elements = {
                'videoType': self.ui.videoTypeComboBox,
                'trackingMetric': self.ui.trackingMetricComboBox,
                'trackingMethod': self.ui.trackingMethodComboBox,
                'numberOfTracker': self.ui.numberOfTrackerComboBox,
                'points': self.ui.pointsComboBox,
                'additionalPoints': self.ui.additionalPointsComboBox,
                'processingSpeed': self.ui.processingSpeedComboBox,
                'topPointOffset': self.ui.topPointOffsetSpinBox,
                'bottomPointOffset': self.ui.bottomPointOffsetSpinBox,
            }


    def show(self):
        """ Show settings dialog """
        self.form.show()


    def __load_settings(self):
        if not os.path.exists(self.settings_file):
            return

        with open(self.settings_file, "r") as f:
            settings = json.load(f)
            for key in self.dialog_elements.keys():
                if key not in settings.keys():
                    continue

                if isinstance(self.dialog_elements[key], QtWidgets.QComboBox):
                    index = self.dialog_elements[key].findText(str(settings[key]), QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.dialog_elements[key].setCurrentIndex(index)
                    else:
                        print("ERROR: Setting not found", str(settings[key]))
                elif isinstance(self.dialog_elements[key], QtWidgets.QSpinBox):
                    self.dialog_elements[key].setValue(0) # always trigger the change event
                    self.dialog_elements[key].setValue(int(settings[key]))
                else:
                    raise NotImplementedError(str(type(self.dialog_elements[key])) + " type is not implemented")


    def __save_settings(self):
        settings = {}
        for key in self.dialog_elements.keys():
            if isinstance(self.dialog_elements[key], QtWidgets.QComboBox):
                settings[key] = self.dialog_elements[key].currentText()
            elif isinstance(self.dialog_elements[key], QtWidgets.QSpinBox):
                settings[key] = self.dialog_elements[key].value()
            else:
                raise NotImplementedError(str(type(self.dialog_elements[key])) + " type is not implemented")

        with open(self.settings_file, "w") as f:
            json.dump(settings, f)


    def __setup_ui_bindings(self):
        self.ui.okButton.clicked.connect(self.__apply)
        self.ui.videoTypeComboBox.currentTextChanged.connect(
                lambda value: self.__set_str_setting(
                    'videoType',
                    list(filter(lambda x: PROJECTION[x]['name'] == value, PROJECTION.keys()))[0]
                )
            )
        self.ui.trackingMetricComboBox.currentTextChanged.connect(self.__set_tracking_metric)
        self.ui.trackingMethodComboBox.currentTextChanged.connect(lambda value: self.__set_str_setting('trackingMethod', value))
        self.ui.numberOfTrackerComboBox.currentTextChanged.connect(lambda value: self.__set_str_setting('numberOfTrackers', value))
        self.ui.pointsComboBox.currentTextChanged.connect(lambda value: self.__set_str_setting('points', value))
        self.ui.additionalPointsComboBox.currentTextChanged.connect(lambda value: self.__set_str_setting('additionalPoints', value))
        self.ui.processingSpeedComboBox.currentTextChanged.connect(lambda value: self.__set_str_setting('skipFrames', value))
        self.ui.topPointOffsetSpinBox.valueChanged.connect(lambda value: self.__set_number_setting("topPointOffset", value))
        self.ui.bottomPointOffsetSpinBox.valueChanged.connect(lambda value: self.__set_number_setting("bottomPointOffset", value))


    def __setup_combo_boxes(self):
        self.ui.videoTypeComboBox.addItems([PROJECTION[key]['name'] \
                for key in PROJECTION.keys() \
                if 'vr' not in key.lower() or self.include_vr])
        self.ui.trackingMethodComboBox.addItems(['Unsupervised Woman', 'Unsupervised Woman + Men', 'Supervised Woman', 'Supervised Woman + Men']) # set before tracking metric
        self.ui.trackingMetricComboBox.addItems(['y (up-down)', 'y inverted (down-up)', 'x (left-right)', 'x inverted (right-left)', 'distance (p1-p2)', 'distance inverted (p2-p1)', 'roll (rotation)', 'roll inverted (rotation)'])
        self.ui.pointsComboBox.addItems(["Direction Changed", "Local Min Max"])
        self.ui.additionalPointsComboBox.addItems(["None", "High Second Derivative", "Distance Minimization"])
        self.ui.processingSpeedComboBox.addItems(["0 (accurate)", "1 (normal)", "2 (fast)"])

        self.ui.numberOfTrackerComboBox.addItems([str(i) for i in range(1, 6)])


    def __set_tracking_metric(self, value):
        value = value.split('(')[0].strip()
        current_tracking_method_items = [self.ui.trackingMethodComboBox.itemText(i) for i in range(self.ui.trackingMethodComboBox.count())]

        if value in ['x', 'y', 'x inverted', 'y inverted']:
            if 'Unsupervised Woman' not in current_tracking_method_items:
                self.ui.trackingMethodComboBox.addItems(['Unsupervised Woman', 'Supervised Woman'])
        else:
            if 'Unsupervised Woman' in current_tracking_method_items:
                self.ui.trackingMethodComboBox.clear()
                self.ui.trackingMethodComboBox.addItems(['Unsupervised Woman + Men', 'Supervised Woman + Men'])

        self.__set_str_setting('trackingMetric', value)


    def __apply(self):
        self.__save_settings()
        self.form.hide()
        self.applySettings.emit()


    def __set_str_setting(self, key, value):
        value = value.split('(')[0].strip()
        self.settings[key] = value

    def __set_number_setting(self, key, value):
        self.settings[key] = value
