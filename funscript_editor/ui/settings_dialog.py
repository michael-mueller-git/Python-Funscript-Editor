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
        self.form.setWindowTitle("Settings")
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
                'videoType': {
                    'instance': self.ui.videoTypeComboBox,
                    'type': 'combobox'
                },
                'trackingMetric': {
                    'instance': self.ui.trackingMetricComboBox,
                    'type': 'combobox'
                },
                'trackingMethod': {
                    'instance': self.ui.trackingMethodComboBox,
                    'type': 'combobox'
                },
                'numberOfTracker': {
                    'instance': self.ui.numberOfTrackerComboBox,
                    'type': 'combobox'
                }
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

                if self.dialog_elements[key]['type'] == 'combobox':
                    index = self.dialog_elements[key]['instance'].findText(str(settings[key]), QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.dialog_elements[key]['instance'].setCurrentIndex(index)
                    else:
                        print("ERROR: Setting not found", str(settings[key]))
                else:
                    raise NotImplementedError(self.dialog_elements[key]['type'] + "type is not implemented")


    def __save_settings(self):
        settings = {}
        for key in self.dialog_elements.keys():
            if self.dialog_elements[key]['type'] == 'combobox':
                settings[key] = self.dialog_elements[key]['instance'].currentText()
            else:
                raise NotImplementedError(self.dialog_elements[key]['type'] + "type is not implemented")

        with open(self.settings_file, "w") as f:
            json.dump(settings, f)


    def __setup_ui_bindings(self):
        self.ui.okButton.clicked.connect(self.__apply)
        self.ui.videoTypeComboBox.currentTextChanged.connect(
                lambda value: self.__set_setting(
                    'videoType',
                    list(filter(lambda x: PROJECTION[x]['name'] == value, PROJECTION.keys()))[0]
                )
            )
        self.ui.trackingMetricComboBox.currentTextChanged.connect(self.__set_tracking_metric)
        self.ui.trackingMethodComboBox.currentTextChanged.connect(lambda value: self.__set_setting('trackingMethod', value))
        self.ui.numberOfTrackerComboBox.currentTextChanged.connect(lambda value: self.__set_setting('numberOfTrackers', value))


    def __setup_combo_boxes(self):
        self.ui.videoTypeComboBox.addItems([PROJECTION[key]['name'] \
                for key in PROJECTION.keys() \
                if 'vr' not in key.lower() or self.include_vr])
        self.ui.trackingMethodComboBox.addItems(['Unsupervised Woman', 'Unsupervised Woman + Men', 'Supervised Woman', 'Supervised Woman + Men']) # set before tracking metric
        self.ui.trackingMetricComboBox.addItems(['y (up-down)', 'y inverted (down-up)', 'x (left-right)', 'x inverted (right-left)', 'distance (p1-p2)', 'distance inverted (p2-p1)', 'roll (rotation)', 'roll inverted (rotation)'])

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

        self.__set_setting('trackingMetric', value)


    def __apply(self):
        self.__save_settings()
        self.form.hide()
        self.applySettings.emit()


    def __set_setting(self, key, value):
        self.settings[key] = value
