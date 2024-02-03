""" Settings Dialog for the Funscript Generator """
import json
import os
import sys
import webbrowser

import funscript_editor.ui.settings_view as settings_view
import funscript_editor.definitions as definitions
import funscript_editor.utils.config as config

from funscript_editor.utils.config import PROJECTION
from funscript_editor.definitions import CONFIG_DIR

from PyQt5 import QtWidgets, QtCore, QtGui


class MyQDialog(QtWidgets.QDialog):

    def __init__(self):
        super(MyQDialog, self).__init__()
        self.close_with_ok = False

    def closeEvent(self, event):
        event.accept()
        if not self.close_with_ok:
            sys.exit(1)


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
        self.form = MyQDialog()
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
                'processingSpeed': self.ui.processingSpeedComboBox,
                'outputMode': self.ui.outputComboBox
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
        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f)
        except:
            print("Save dialog settings FAILED")


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
                "y + roll (up-down + rotation)",
                "distance + roll (p1-p1 + rotation)",
                "x + y (left-right + up-down)",
                "x + y + roll (left-right + up-down + rotation)",
            ])

        self.ui.processingSpeedComboBox.addItems([
            "0 (accurate)",
            "1 (normal)",
            "2 (fast)"
        ])

        self.ui.outputComboBox.addItems([
            "post processed data",
            "normalized raw tracking data"
        ])

        self.ui.numberOfTrackerComboBox.addItems([str(i) for i in range(1, 6)])


    def __set_tracking_metric(self, value):
        value = value.split('(')[0].strip()
        selection = self.ui.trackingMethodComboBox.currentText()

        self.ui.trackingMethodComboBox.clear()
        if value in ['x', 'y', 'x inverted', 'y inverted', 'x + y']:
            self.ui.trackingMethodComboBox.addItems(self.trackingMethods)
        else:
            self.ui.trackingMethodComboBox.addItems(list(filter(lambda x: "one" not in x, self.trackingMethods)))

        index = self.ui.trackingMethodComboBox.findText(selection, QtCore.Qt.MatchFixedString)
        if index >= 0:
                self.ui.trackingMethodComboBox.setCurrentIndex(index)


    def __apply(self):
        self.form.close_with_ok = True
        self.__save_settings()
        self.form.hide()
        self.applySettings.emit()


    def __set_str_setting(self, key, value):
        value = value.split('(')[0].strip()
        self.settings[key] = value

    def __set_number_setting(self, key, value):
        self.settings[key] = value
