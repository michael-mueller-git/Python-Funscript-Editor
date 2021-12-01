import bisect
import numpy as np
import logging

from dataclasses import dataclass
from numpy.linalg import norm

@dataclass
class SignalParameter:
    avg_sec_for_local_min_max_extraction: float = 2.0

class Signal:

    def __init__(self, fps):
        self.params = SignalParameter()
        self.fps = fps
        self.logger = logging.getLogger(__name__)


    @staticmethod
    def moving_average(x: list, w: int) -> list:
        """ Calculate moving average for given signal x with window size w

        Args:
            x (list): list with float or int signal values
            w (int): window size

        Returns:
            list: moving average for signal x
        """
        w = round(w)

        if len(x) == 0:
            return []

        if len(x) <= w+1:
            return[np.mean(x) for _ in range(len(x))]

        if w == 1:
            return x

        avg = np.convolve(x, np.ones(int(w*2)), 'valid') / int(w*2)
        return [avg[0] for _ in range(int(w))] \
                + list(avg) \
                + [avg[-1] for _ in range(len(avg)+w,len(x))]


    @staticmethod
    def moving_standard_deviation(x: list, w: int) -> list:
        """ Get 2 seconds moving standard deviation of given signal

        Args:
            x (list): input signal
            w (int): window size

        Returns:
            list: moving standard deviation
        """
        w = round(w)

        if len(x) == 0:
            return []

        if len(x)-w <= w:
            return[np.std(x) for _ in range(len(x))]

        std = [np.std(x[ii-w:ii+w+1]) for ii in range(w,len(x)-w)]
        return [std[0] for _ in range(w)] \
                + list(std) \
                + [std[-1] for _ in range(w)]


    @staticmethod
    def first_derivative(x: list, w: int = 1) -> list:
        """ Get first derivative of given signal

        Args:
            x (list): input signal
            w (int): window size for signal smothing

        Returns:
            list: first derivative
        """
        if w < 0:
            w = 1

        if w % 2 == 0:
            w += 1

        return Signal.moving_average(np.diff(x, 1).tolist(), w)


    @staticmethod
    def second_derivative(x: list, w: int = 1) -> list:
        """ Get second derivative of given signal

        Args:
            x (list): input signal
            w (int): window size for signal smothing

        Returns:
            list: second derivative
        """
        if w < 0:
            w = 1

        if w % 2 == 0:
            w += 1

        return Signal.moving_average(np.diff(Signal.moving_average(np.diff(x, 1).tolist(), w), 1).tolist(), w)


    @staticmethod
    def scale_signal(signal: list, lower: float = 0, upper: float = 99) -> list:
        """ Scale an signal (list of float or int) between given lower and upper value

        Args:
            signal (list): list with float or int signal values to scale
            lower (float): lower scale value
            upper (float): upper scale value

        Returns:
            list: list with scaled signal
        """
        if len(signal) == 0:
            return signal

        if len(signal) == 1:
            return [lower]

        signal_min = min(signal)
        signal_max = max(signal)
        return [(float(upper) - float(lower)) * (x - signal_min) / (signal_max - signal_min) + float(lower) for x in signal]


    @staticmethod
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
        if len(signal) == 0:
            return signal

        if len(signal) == 1:
            return [lower]

        a1 = np.quantile(signal, lower_quantile)
        a2 = np.quantile(signal, upper_quantile)
        anomaly_free = np.array([x for x in signal if a1 < x < a2])
        anomaly_free_min = min(anomaly_free)
        anomaly_free_max = max(anomaly_free)
        scaled = [(upper - lower) * (x - anomaly_free_min) / (anomaly_free_max - anomaly_free_min) + lower for x in signal]
        return [min((anomaly_free_max, max((anomaly_free_min, x)) )) for x in scaled]


    @staticmethod
    def find_nearest(array: list, value: float, side: str) -> float:
        """ Find nearest value in SORTED array

        Args:
            array (list): sorted list with values
            value (float): search value
            side (str): 'left' or 'right'

        Returns:
            float: nearest value in array
        """
        if side.lower() == 'left':
            pos = 0
            for i in range(len(array)):
                if value <= array[i]:
                    break
                pos = i

            return array[pos]

        elif side.lower() == 'right':
            pos = 0
            for i in reversed(range(len(array))):
                if value >= array[i]:
                    break
                pos = i

            return array[pos]
        else:
            raise NotImplementedError("find_nearest is not implemented for side={}".format(side))


    def get_high_second_derivative_points(self, score: list, alpha: float) -> list:
        """ Get change points by comparing second derivative with the rolling standard deviation

        Args:
            score (list): list with float or int signal values
            alpha (float): threshold value for standard deviation comparing

        Returns:
            list: idx list with changepoints
        """
        dx2 = Signal.second_derivative(score)
        dx2_abs = abs(np.array(dx2))
        std = Signal.moving_standard_deviation(dx2, round(self.fps * self.params.avg_sec_for_local_min_max_extraction))
        changepoints, tmp_max_idx = [], -1
        for pos in range(len(dx2_abs)):
            if abs(dx2_abs[pos]) > alpha*std[pos]:
                if tmp_max_idx < 0 or dx2_abs[tmp_max_idx] <= dx2_abs[pos]:
                    tmp_max_idx = pos
            elif tmp_max_idx >= 0:
                changepoints.append(tmp_max_idx)
                tmp_max_idx = -1

        return changepoints


    def get_edge_points(self, score: list, changepoints: list, threshold: float = 25.0) -> list:
        """ Get Edge Points by calculate the distance to each point in the score.

        Note:
            We map the time axis between each predicted changepoint to min_pos - max_pos in
            this section to get usable distances.

        Args:
            score (list): the predicted score
            changepoints (list): current changepoints
            threshold (float): threshold value to predict additional edge point

        Returns:
            list: list with index of the edge points (additional change points)
        """
        edge_points, overall_max_distance = [], 0
        changepoints.sort()
        if len(changepoints) < 2: return []
        for i in range(len(changepoints)-1):
            min_pos = min([score[changepoints[i]], score[changepoints[i+1]]])
            max_pos = max([score[changepoints[i]], score[changepoints[i+1]]])
            start = np.array([min_pos, score[changepoints[i]]])
            end = np.array([max_pos, score[changepoints[i+1]]])
            scale = lambda x: ((max_pos - min_pos) * (x - changepoints[i]) / float(changepoints[i+1] - changepoints[i]) + float(min_pos))
            distances = [ norm(np.cross(end-start, start-np.array([scale(j), score[j]])))/norm(end-start) \
                    for j in range(changepoints[i], changepoints[i+1]) ]
            max_distance = max(distances)
            if overall_max_distance < max_distance:
                overall_max_distance = max_distance
            if max_distance > threshold:
                edge_points.append(changepoints[i] + distances.index(max_distance))

        self.logger.info("Max distance was {}".format(overall_max_distance))
        return edge_points



    def get_local_max_and_min_idx(self, score :list) -> list:
        """ Get the local max and min positions in given signal

        Args:
            score (list): list with float signal

        Returns:
            list:  with local max and min indexes
        """
        avg = Signal.moving_average(score, w=round(self.fps * self.params.avg_sec_for_local_min_max_extraction))
        smothed_score = Signal.moving_average(score, w=3)
        changepoints, tmp_min_idx, tmp_max_idx = [], -1, -1
        for pos in range(len(smothed_score)):
            if smothed_score[pos] < avg[pos]:
                if tmp_min_idx < 0:
                    tmp_min_idx = pos
                elif smothed_score[tmp_min_idx] >= smothed_score[pos]:
                    tmp_min_idx = pos
            elif tmp_min_idx >= 0:
                changepoints.append(tmp_min_idx)
                tmp_min_idx = -1

            if smothed_score[pos] > avg[pos]:
                if tmp_max_idx < 0:
                    tmp_max_idx = pos
                elif smothed_score[tmp_max_idx] <= smothed_score[pos]:
                    tmp_max_idx = pos
            elif tmp_max_idx >= 0:
                changepoints.append(tmp_max_idx)
                tmp_max_idx = -1

        return changepoints


    @staticmethod
    def get_direction_changes(signal: list, filter_len: int = 3) -> list:
        """ Get direction changes positions in given signal

        Args:
            signal (list): list with float signal
            filter_len (int): length of the filter to detect an direction change

        Returns:
            list:  indexes of direction changes
        """
        if len(signal) < filter_len:
            return []

        filtered_indexe = [i for i in range(len(signal)-filter_len) \
                if all(signal[i+j] > signal[i+j+1] for j in range(filter_len)) \
                or all(signal[i+j] < signal[i+j+1] for j in range(filter_len))
        ]

        if len(filtered_indexe) < 2:
            return []

        direction = [-1 if signal[filtered_indexe[i]] > signal[filtered_indexe[i+1]] else 1 \
                for i in range(len(filtered_indexe) - 1)]

        changepoints, start_position, current_direction = [], min(filtered_indexe), direction[0]
        for idx, direction in zip(filtered_indexe, direction):
            if direction != current_direction:
                changepoints.append(idx + start_position)
                current_direction = direction

        return changepoints
