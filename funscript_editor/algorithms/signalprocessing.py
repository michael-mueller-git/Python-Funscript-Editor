""" Signal Processing Algorithms """

import numpy as np
from funscript_editor.utils.config import HYPERPARAMETER

def scale_signal(signal :list, lower: float = 0, upper: float = 99) -> list:
    """ Scale an signal (list of float or int) between given lower and upper value

    Args:
        signal (list): list with float or int signal values to scale
        lower (float): lower scale value
        upper (float): upper scale value

    Returns:
        list: list with scaled signal
    """
    return [(upper - lower) * (x - min(signal)) / (max(signal) - min(signal)) + lower for x in signal]


def moving_average(x :list, w: int) -> list:
    """ Calculate moving average for given signal x with window size w

    Args:
        x (list): list with float or int signal values
        w (int): window size

    Returns:
        list: moving average for signal x
    """
    w = round(w)
    avg = np.convolve(x, np.ones(int(w*2)), 'valid') / int(w*2)
    # XXX use extrapolation function
    return [avg[0] for _ in range(int(w))]\
            +list(avg)\
            +[avg[-1] for _ in range(len(avg)+w,len(x))]


def get_local_max_and_min_idx(score :list, fps: int) -> list:
    """ Get the local max and min positions in given signal

    Args:
        score (list): list with float or int signal values
        fps (int): rounded fps of the video

    Returns:
        list: list with all local max and min indexes
    """
    avg = moving_average(score, w=round(fps * HYPERPARAMETER['avg_sec_for_local_min_max_extraction']))
    tmp_min_idx, tmp_max_idx, idx_list = -1, -1, []
    for pos in range(len(score)):
        if score[pos] < avg[pos]:
            if tmp_min_idx < 0: tmp_min_idx = pos
            elif score[tmp_min_idx] > score[pos]: tmp_min_idx = pos
        elif tmp_min_idx >= 0:
            idx_list.append(tmp_min_idx)
            tmp_min_idx = -1

        if score[pos] > avg[pos]:
            if tmp_max_idx < 0: tmp_max_idx = pos
            elif score[tmp_max_idx] < score[pos]: tmp_max_idx = pos
        elif tmp_max_idx >= 0:
            idx_list.append(tmp_max_idx)
            tmp_max_idx = -1
    return idx_list
