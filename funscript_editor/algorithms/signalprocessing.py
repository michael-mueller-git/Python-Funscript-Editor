""" Signal Processing Algorithms """

import numpy as np
import logging
import platform
from funscript_editor.utils.config import HYPERPARAMETER, SETTINGS
from numpy.linalg import norm

def scale_signal(signal :list, lower: float = 0, upper: float = 99) -> list:
    """ Scale an signal (list of float or int) between given lower and upper value

    Args:
        signal (list): list with float or int signal values to scale
        lower (float): lower scale value
        upper (float): upper scale value

    Returns:
        list: list with scaled signal
    """
    if len(signal) == 0: return signal
    if len(signal) == 1: return [lower]
    signal_min = min(signal)
    signal_max = max(signal)
    return [(float(upper) - float(lower)) * (x - signal_min) / (signal_max - signal_min) + float(lower) for x in signal]


def scale_signal_with_anomalies(
        signal :list,
        lower: float = 0,
        upper: float = 99,
        lower_quantile: float = 0.0005,
        upper_quantile: float = 0.9995) -> list:
    """ Scale an signal (list of float or int) between given lower and upper value

    Args:
        signal (list): list with float or int signal values to scale
        lower (float): lower scale value
        upper (float): upper scale value
        lower_quantile (float): lower quantile value to filter [0,1]
        upper_quantile (float): upper quantile value to filter [0,1]

    Returns:
        list: list with scaled signal
    """
    if len(signal) == 0: return signal
    if len(signal) == 1: return [lower]
    a1 = np.quantile(signal, lower_quantile)
    a2 = np.quantile(signal, upper_quantile)
    anomaly_free = np.array([x for x in signal if a1 < x < a2])
    anomaly_free_min = min(anomaly_free)
    anomaly_free_max = max(anomaly_free)
    scaled = [(upper - lower) * (x - anomaly_free_min) / (anomaly_free_max - anomaly_free_min) + lower for x in signal]
    return [min((anomaly_free_max, max((anomaly_free_min, x)) )) for x in scaled]


def moving_average(x :list, w: int) -> list:
    """ Calculate moving average for given signal x with window size w

    Args:
        x (list): list with float or int signal values
        w (int): window size

    Returns:
        list: moving average for signal x
    """
    w = round(w)
    if len(x) == 0: return []
    if len(x) <= w+1: return[np.mean(x) for _ in range(len(x))]
    avg = np.convolve(x, np.ones(int(w*2)), 'valid') / int(w*2)
    # TODO use extrapolation function
    return [avg[0] for _ in range(int(w))]\
            +list(avg)\
            +[avg[-1] for _ in range(len(avg)+w,len(x))]


def moving_standard_deviation(x: list, w: int) -> list:
    """ Get 2 seconds moving standard deviation of given signal

    Args:
        x (list): input signal
        w (int): window size

    Returns:
        list: moving standard deviation
    """
    w = round(w)
    if len(x) == 0: return []
    if len(x)-w <= w: return[np.std(x) for _ in range(len(x))]
    std = [np.std(x[ii-w:ii+w+1]) for ii in range(w,len(x)-w)]
    # TODO use extrapolation function
    return [std[0] for _ in range(w)]\
            +list(std)\
            +[std[-1] for _ in range(w)]


def second_derivative(x: list, w: int = 1) -> list:
    """ Get second derivative of given signal

    Args:
        x (list): input signal
        w (int): window size for signal smothing

    Returns:
        list: second derivative
    """
    if w < 0: w = 1
    if w % 2 == 0: w += 1
    return moving_average(np.diff(moving_average(np.diff(x, 1).tolist(), w), 1).tolist(), w)


def get_changepoints(score: list, fps: int, alpha: float) -> list:
    """ Get change points by comparing second derivative with the rolling standard deviation

    Args:
        score (list): list with float or int signal values
        fps (int): rounded fps of the video
        alpha (float): threshold value for standard deviation comparing

    Returns:
        list: idx list with changepoints
    """
    dx2 = second_derivative(score)
    std = moving_standard_deviation(dx2, round(fps * HYPERPARAMETER['avg_sec_for_local_min_max_extraction']))
    dx2_abs = abs(np.array(dx2))
    tmp_max_idx = -1
    changepoints = []
    for pos in range(len(dx2_abs)):
        if abs(dx2_abs[pos]) > alpha*std[pos]:
            if tmp_max_idx < 0: tmp_max_idx = pos
            elif dx2_abs[tmp_max_idx] <= dx2_abs[pos]: tmp_max_idx = pos
        elif tmp_max_idx >= 0:
            changepoints.append(tmp_max_idx)
            tmp_max_idx = -1

    #if tmp_max_idx > 0:
    #    changepoints.append((tmp_max_idx))

    return changepoints


def get_edge_points(score: list, changepoints: dict, threshold: float = 150.0) -> list:
    """ Get Edge Points by calculate the distance to each point in the score.

    Note:
        We map the time axis between each predicted changepoint to min_pos - max_pos in
        this section to get usable distances.

    Args:
        score (list): the predicted score
        changepoints (dict): current min and max changepoints
        threshold (float): threshold value to predict additional edge point

    Returns:
        list: list with index of the edge points (additional change points)
    """
    edge_points = []
    overall_max_distance = 0
    cp = changepoints['min']+changepoints['max']
    cp.sort()
    if len(cp) < 2: return []
    for i in range(len(cp)-1):
        min_pos = min([score[cp[i]], score[cp[i+1]]])
        max_pos = max([score[cp[i]], score[cp[i+1]]])
        start = np.array([min_pos, score[cp[i]]])
        end = np.array([max_pos, score[cp[i]]])
        scale = lambda x: ((max_pos - min_pos) * (x - cp[i]) / float(cp[i+1] - cp[i]) + float(min_pos))
        distances = [ norm(np.cross(end-start, start-np.array([scale(j), score[j]])))/norm(end-start) \
                for j in range(cp[i], cp[i+1]) ]
        max_distance = max(distances)
        if overall_max_distance < max_distance:
            overall_max_distance = max_distance
        if max_distance > threshold:
            print("Add Edge point for distance", max_distance)
            edge_points.append(cp[i] + distances.index(max_distance))

    print("Max distance was", overall_max_distance)
    return edge_points


def get_local_max_and_min_idx(score :list, fps: int, shift_min :int = 0, shift_max :int = 0) -> dict:
    """ Get the local max and min positions in given signal

    Args:
        score (list): list with float or int signal values
        fps (int): rounded fps of the video
        shift_min (int): shift the local min indexes by given value (default is 0)
        shift_max (int): shift the local max indexes by given value (default is 0)

    Returns:
        dict: dict with 2 lists with all local max and min indexes ({'min':[], 'max':[]})
    """
    if platform.system() != 'Windows':
        # TODO logging here on windows cause open background process
        logger = logging.getLogger("changepoints")

    avg = moving_average(score, w=round(fps * HYPERPARAMETER['avg_sec_for_local_min_max_extraction']))
    changepoints = {'min': [], 'max': []}
    tmp_min_idx, tmp_max_idx = -1, -1
    for pos in range(len(score)):
        if score[pos] < avg[pos]:
            if tmp_min_idx < 0: tmp_min_idx = pos
            elif score[tmp_min_idx] >= score[pos]: tmp_min_idx = pos
        elif tmp_min_idx >= 0:
            changepoints['min'].append(tmp_min_idx)
            tmp_min_idx = -1

        if score[pos] > avg[pos]:
            if tmp_max_idx < 0: tmp_max_idx = pos
            elif score[tmp_max_idx] <= score[pos]: tmp_max_idx = pos
        elif tmp_max_idx >= 0:
            changepoints['max'].append(tmp_max_idx)
            tmp_max_idx = -1

    #if tmp_min_idx > 0:
    #    changepoints['min'].append((tmp_min_idx))

    #if tmp_max_idx > 0:
    #    changepoints['max'].append((tmp_max_idx))

    delta_max = (max(score) - min(score)) * 0.01 * min((10.0, float(HYPERPARAMETER['local_max_delta_in_percent'])))
    delta_min = (max(score) - min(score)) * 0.01 * min((10.0, float(HYPERPARAMETER['local_min_delta_in_percent'])))

    # shift points to the real change points
    for k, idx in enumerate(changepoints['min']):
        new_pos = idx
        while new_pos+1 < len(score) and score[idx] + delta_min > score[new_pos+1]:
            new_pos += 1
        changepoints['min'][k] = new_pos

    for k, idx in enumerate(changepoints['max']):
        new_pos = idx
        while new_pos+1 < len(score) and score[idx] - delta_max < score[new_pos+1]:
            new_pos += 1
        changepoints['max'][k] = new_pos


    if SETTINGS['additional_changepoints']:
        if platform.system() != 'Windows':
            # TODO logging here on windows cause open background process
            logger.info("Add additional change points")
        merge_threshold = max(1, round(fps * float(HYPERPARAMETER['additional_changepoints_merge_threshold_in_ms']) / 1000.0))
        additional_changepoints = get_changepoints(score, fps, float(HYPERPARAMETER['changepoint_detection_threshold']))
        for cp_idx in additional_changepoints:
            if len(list(filter(lambda x: abs(cp_idx - x) <= merge_threshold, changepoints['min']))) > 0:
                continue

            if len(list(filter(lambda x: abs(cp_idx - x) <= merge_threshold, changepoints['max']))) > 0:
                continue

            if score[cp_idx] < avg[cp_idx]:
                changepoints['min'].append(cp_idx)
            else:
                changepoints['max'].append(cp_idx)


    if platform.system() != 'Windows':
        # TODO logging here on windows cause open background process
        logger.info("Apply manual shift")
    if shift_min != 0:
        for k, idx in enumerate(changepoints['min']):
            changepoints['min'][k] = max((0, min((len(score)-1, idx+shift_min)) ))

    if shift_max != 0:
        for k, idx in enumerate(changepoints['max']):
            changepoints['max'][k] = max((0, min((len(score)-1, idx+shift_max)) ))

    return changepoints
