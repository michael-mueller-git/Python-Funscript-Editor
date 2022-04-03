""" Settings Dialog for the Funscript Generator """
import json
import os
import webbrowser

import funscript_editor.ui.settings_view as settings_view
import funscript_editor.definitions as definitions
import funscript_editor.utils.config as config

from funscript_editor.utils.config import PROJECTION
from funscript_editor.definitions import CONFIG_DIR

from PyQt5 import QtWidgets, QtCore, QtGui

class SettingsDialog(QtWidgets.QDialog):
    """ Settings Dialog

    Args:
        settings (dict): dict where to store the settings
        include_vr (bool): include vr options
        include_multiaxis (bool): include multiaxis output
    """

    def __init__(self, settings: dict, include_vr: bool = True, include_multiaxis : bool = True):
        super(SettingsDialog, self).__init__()
        self.include_vr = include_vr
        self.include_multiaxis = include_multiaxis
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
                'trackingMetrics': self.ui.trackingMetricComboBox,
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
                self.__set_str_setting(key, settings[key])
            elif isinstance(self.dialog_elements[key], QtWidgets.QSpinBox):
                settings[key] = self.dialog_elements[key].value()
                self.__set_number_setting(key, settings[key])
            else:
                raise NotImplementedError(str(type(self.dialog_elements[key])) + " type is not implemented")

        with open(self.settings_file, "w") as f:
            json.dump(settings, f)


    def __setup_ui_bindings(self):
        self.ui.okButton.clicked.connect(self.__apply)
        self.ui.trackingMetricComboBox.currentTextChanged.connect(self.__set_tracking_metric)
        self.ui.docsButton.clicked.connect(self.__open_documentation)


    def __open_documentation(self):
        try:
            browser = webbrowser.get()
            browser.open_new(definitions.DOCS_URL.format(tag=str('main' if config.VERSION == '0.0.0' else 'v'+config.VERSION)))
        except:
            pass


    def __setup_combo_boxes(self):
        self.ui.videoTypeComboBox.addItems([PROJECTION[key]['name'] \
                for key in PROJECTION.keys() \
                if 'vr' not in key.lower() or self.include_vr])

        self.trackingMethods = [
            'Unsupervised one moving person',
            'Unsupervised two moving persons',
            'Supervised stopping one moving person',
            'Supervised stopping two moving persons',
            'Supervised ignoring one moving person',
            'Supervised ignoring two moving persons'
            ]

        self.ui.trackingMethodComboBox.addItems(self.trackingMethods) # set before tracking metric!

        self.ui.trackingMetricComboBox.addItems([
            'y (up-down)',
            'y inverted (down-up)',
            'x (left-right)',
            'x inverted (right-left)',
            'distance (p1-p2)',
            'distance inverted (p2-p1)',
            'roll (rotation)',
            'roll inverted (rotation)'
        ])

        if self.include_multiaxis:
            self.ui.trackingMetricComboBox.addItems([
                "y + roll (up-down + rotation)"
            ])

        self.ui.pointsComboBox.addItems([
            "Local Min Max",
            "Direction Changed"
        ])

        self.ui.additionalPointsComboBox.addItems([
            "None",
            "High Second Derivative",
            "Distance Minimization"
        ])

        self.ui.processingSpeedComboBox.addItems([
            "0 (accurate)",
            "1 (normal)",
            "2 (fast)"
        ])

        self.ui.numberOfTrackerComboBox.addItems([str(i) for i in range(1, 6)])


    def __set_tracking_metric(self, value):
        value = value.split('(')[0].strip()
        selection = self.ui.trackingMethodComboBox.currentText()

        self.ui.trackingMethodComboBox.clear()
        if value in ['x', 'y', 'x inverted', 'y inverted']:
            self.ui.trackingMethodComboBox.addItems(self.trackingMethods)
        else:
            self.ui.trackingMethodComboBox.addItems(list(filter(lambda x: "one" not in x, self.trackingMethods)))

        index = self.ui.trackingMethodComboBox.findText(selection, QtCore.Qt.MatchFixedString)
        if index >= 0:
                self.ui.trackingMethodComboBox.setCurrentIndex(index)


    def __apply(self):
        self.__save_settings()
        self.form.hide()
        self.applySettings.emit()


    def __set_str_setting(self, key, value):
        value = value.split('(')[0].strip()
        self.settings[key] = value

    def __set_number_setting(self, key, value):
        self.settings[key] = value
