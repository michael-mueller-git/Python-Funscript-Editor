""" Settings Dialog for the Funscript Generator """

import funscript_editor.ui.settings_view as settings_view

from funscript_editor.utils.config import PROJECTION

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
        self.__setup_ui_bindings()
        self.__setup_combo_boxes()

    #: apply settings event
    applySettings = QtCore.pyqtSignal()

    def show(self):
        """ Show wttings dialog """
        self.form.show()

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

    def __setup_combo_boxes(self):
        self.ui.videoTypeComboBox.addItems([PROJECTION[key]['name'] \
                for key in PROJECTION.keys() \
                if 'vr' not in key.lower() or self.include_vr])
        self.ui.trackingMetricComboBox.addItems(['y (up-down)', 'x (left-right)', 'euclideanDistance', 'pitch'])
        self.ui.trackingMethodComboBox.addItems(['Woman', 'Woman + Men'])

    def __set_tracking_metric(self, value):
        value = value.split('(')[0].strip()
        if value in ['x', 'y']:
            self.ui.trackingMethodComboBox.setEnabled(True)
        else:
            self.ui.trackingMethodComboBox.setCurrentText('Woman + Men')
            self.ui.trackingMethodComboBox.setEnabled(False)

        self.__set_setting('trackingMetric', value)

    def __apply(self):
        self.form.hide()
        self.applySettings.emit()

    def __set_setting(self, key, value):
        self.settings[key] = value
