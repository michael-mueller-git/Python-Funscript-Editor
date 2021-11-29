import bisect
import numpy as np
import logging

from dataclasses import dataclass
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS
from numpy.linalg import norm

@dataclass
class DecimateParameter:
    test = 1

class Decimate:

    def __init__(self, params: DecimateParameter, signal: list):
        self.params = params
        self.raw_signal = signal
        self.logger = logging.getLogger(__name__)


    def compute(self):
        pass
