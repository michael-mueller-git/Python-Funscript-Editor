""" Funscript Dataclass

Funscript is Funjack's haptic script format. It's basically JSON encoded timed positions:

.. code-block:: python

    {
        "version": "1.0",
        "inverted": false,
        "range": 90,
        "actions": [
                {"pos": 0, "at": 100},
                {"pos": 100, "at": 500},
                ...
        ]
    }

- version: funscript version (default="1.0")
- inverted (bool): positions are inverted, example: 0=100,100=0 (default=false)
- range: range of moment to use in percent (0-100) (default=90)
- actions: script for a Launch
    - pos: position in percent (0-100)
    - at : time to be at position in milliseconds
"""

import os
import cv2
import json

import numpy as np


class Funscript:
    """
    Funscript Class

    Args:
        fps (float): Video FPS
        version (str): funscript version (default="1.0")
        inverted (bool): positions are inverted, example: 0=100,100=0 (default=false)
        moment_range (int): range of moment to use in percent (0-100) (default=90)
    """

    def __init__(self, fps, version='1.0', inverted=False, moment_range=90):
        self.data = {
                'version': version,
                'inverted': inverted,
                'range': max((0, min((moment_range, 100)))),
                'fps': fps,
                'actions': [],
                'sections': []
                }
        self.changed = False


    def get_fps(self) -> float:
        """ Get Video FPS

        Returns:
            float: video FPS
        """
        return self.data['fps']


    def is_empty(self) -> bool:
        """ Check if the funscript actions are empty

        Returns:
            bool: True if no action exist else False
        """
        return len(self.data['actions']) < 1


    def delete_action(self, timestamp: int):
        """ Deleta an action by timestamp

        Args:
            timestamp (int): timestamp in milliseconds

        Returns:
            Funscript: current funscript instance
        """
        del_list = []
        for i in range(len(self.data['actions'])):
            if abs(self.data['actions'][i]['at'] - timestamp ) <= 2 * (1000.0 / self.data['fps']):
                del_list.append(i)

        del_list.sort(reverse=True)
        for x in del_list: del self.data['actions'][x]
        self.changed = True
        return self


    def delete_folowing_actions(self, timestamp: int):
        """ Deleta all folowing actions for given timestamp

        Args:
            timestamp (int): timestamp in milliseconds

        Returns:
            Funscript: current funscript instance
        """
        del_list = []
        for i in range(len(self.data['actions'])):
            if self.data['actions'][i]['at'] > timestamp+2:
                del_list.append(i)

        del_list.sort(reverse=True)
        for x in del_list: del self.data['actions'][x]
        self.changed = True
        return self


    def clear_actions(self):
        """ Clear all actions

        Returns:
            Funscript: current funscript instance
        """
        self.data['actions'] = []
        return self


    def get_last_action_time(self) -> int:
        """ Get time of last action in current funscript

        Returns:
            int: timestamp in milliseconds
        """
        if len(self.data['actions']) == 0: return 0
        return self.data['actions'][-1]['at']


    def get_first_action_time(self) -> int:
        """ Get time of first action in current funscript

        Returns:
            int: timestamp in milliseconds
        """
        if len(self.data['actions']) == 0: return 0
        return self.data['actions'][0]['at']


    def ground_all(self, limit :int = 45):
        """ Set all lower strokes below the limit to zero

        Args:
            limit (int): all lower Strokes below this will be set to zero

        Returns:
            Funscript: current funscript instance
        """
        self.changed = True
        for i in range(1, len(self.data['actions'])-1):
            if self.data['actions'][i]['pos'] < limit \
                    and self.data['actions'][i-1]['pos'] > self.data['actions'][i]['pos'] \
                    and self.data['actions'][i+1]['pos'] > self.data['actions'][i]['pos']: \
                    self.data['actions'][i]['pos'] = 0
        return self


    def get_next_action(self, current_timestamp: int) -> dict:
        """ Get action next to current timestamp

        Args:
            current_timestamp (int): current timestamp in milliseconds

        Returns:
            dict: action dictionary with {'pos', 'at'}
        """
        if len(self.data['actions']) < 1: return {'pos': 0, 'at': 0}
        idx = (np.abs(np.array(self.get_actions_times()) - current_timestamp)).argmin()
        if self.data['actions'][idx]['at'] > current_timestamp + 1: return self.data['actions'][idx]
        elif len(self.data['actions']) == 1: return self.data['actions'][0]
        elif idx+1 < len(self.data['actions']): return self.data['actions'][idx+1]
        else: return self.data['actions'][0]


    def get_prev_action(self, current_timestamp: int) -> dict:
        """ Get previous action to current timestamp

        Args:
            current_timestamp (int): current timestamp in milliseconds

        Returns:
            dict: action dictionary with {'pos', 'at'}
        """
        if len(self.data['actions']) < 2: return {'pos': 0, 'at': 0}
        idx = (np.abs(np.array(self.get_actions_times()) - current_timestamp)).argmin()
        if self.data['actions'][idx]['at'] < current_timestamp - 1: return self.data['actions'][idx]
        elif idx > 0: return self.data['actions'][idx-1]
        else: return self.data['actions'][-1]


    def get_stroke_height(self, current_timestamp: int) -> int:
        """ Get stroke height at given timestamp

        Args:
            current_timestamp (int): current timestamp in milliseconds

        Returns:
            int: stroke height (1-100)
        """
        if len(self.get_actions()) < 2: return 0
        return int(round(abs(self.get_next_action(current_timestamp)['pos'] - self.get_prev_action(current_timestamp)['pos'])))


    def add_action(self, position: int, time: int):
        """ Add a new action to the Funscript

        Args:
            position (int): position in percent (0-100)
            time (int): time to be at position in milliseconds

        Returns:
            Funscript: current funscript instance
        """
        self.changed = True
        self.delete_action(time)
        self.data['actions'].append({'pos': int(round(position)), 'at': time})
        self.data['actions'].sort(key = lambda x: x['at'])
        return self


    def get_actions(self) -> list:
        """ Get all actions from current funscript object

        Returns:
            list: funscript actions
        """
        return self.data['actions']


    def get_stroke_time(self, current_timestamp: int) -> int:
        """ Get stroke duration for given timestamp in milliseconds

        Note:
            measure one periode (down-up-down or up-down-up)

        Args:
            current_timestamp (int): current position in milliseconds

        Returns:
            int: stroke duration in milliseconds
        """
        stroke_times_before = [x for x in self.get_actions_times() if x <= current_timestamp]
        stroke_times_after = [x for x in self.get_actions_times() if x > current_timestamp]
        if len(stroke_times_before) == 0: stroke_times_before = [0]
        if len(stroke_times_after) > 1: return int(round(stroke_times_after[1] - stroke_times_before[-1]))
        elif len(stroke_times_before) > 2: return int(round(stroke_times_before[-1] - stroke_times_before[-3]))
        else: return 0


    def get_all_stroke_times(self) -> list:
        """ Get all stroke duration in this Funscript

        Returns:
            list: list with stroke duration in milliseconds
        """
        action_times = self.get_actions_times()
        if len(action_times) < 2: return []
        return [action_times[i+2] - action_times[i] for i in range(len(action_times)-2)]


    def get_fastest_stroke(self) -> int:
        """ Get the fastest stroke time in current Funscript

        Returns:
            int: fastest stroke time in milliseconds
        """
        times = self.get_all_stroke_times()
        return int(round(min(self.get_all_stroke_times()) if len(times) > 1 else 0))


    def get_slowest_stroke(self) -> int:
        """ Get the slowest stroke time in current Funscript

        Returns:
            int: slowest stroke time in milliseconds
        """
        times = self.get_all_stroke_times()
        return int(round(max(self.get_all_stroke_times()) if len(times) > 1 else 0))


    def get_median_stroke(self) -> int:
        """ Get the median stroke time for current Funscript

        Returns:
            int: median stroke time in milliseconds
        """
        times = self.get_all_stroke_times()
        return int(round(np.median(np.array(times)) if len(times) > 1 else 0))


    def get_actions_positions(self) -> list:
        """ Get all positions from current funscript object

        Returns:
            list: positions
        """
        return [item['pos'] for item in self.get_actions()]


    def get_actions_times(self) -> list:
        """ Get all action times from current funscript object

        Returns:
            list: times in milliseconds
        """
        return [item['at'] for item in self.get_actions()]


    def __millisec_to_frame(self, milliseconds: int) -> int:
        """ Convert milliseconds to frame number

        Args:
            milliseconds (int): time in milliseconds

        Returns:
            int: frame number for given time
        """
        if milliseconds < 0: return 0
        return int(round(float(milliseconds)/(float(1000)/float(self.data['fps']))))


    def get_actions_frames(self) -> list:
        """ Get all actions frame numbers from current funscript object

        Returns:
            list: frame numbers
        """
        return [self.__millisec_to_frame(item['at']) for item in self.get_actions()]


    def invert_actions(self):
        """ Invert all actions in current Funscript

        Returns:
            Funscript: current funscript instance
        """
        for i in range(len(self.data['actions'])):
            self.data['actions'][i]['pos'] = round(100 - self.data['actions'][i]['pos'])
        return self


    def save(self, filename: str, create_backup: bool = True):
        """ Save funscript to file

        Args:
            filename (path): path where to save the funscript
            create_backup (bool): create an additional backup file

        Returns:
            Funscript: current funscript instance
        """
        if not filename.endswith('.json') and not filename.endswith('.funscript'): filename += '.funscript'
        for i in range(len(self.data['actions'])): self.data['actions'][i]['pos'] = round(self.data['actions'][i]['pos'])

        with open(filename, 'w') as json_file: json.dump(self.data, json_file, indent=4)

        if create_backup:
            num=0
            while os.path.exists(filename + str(num)): num += 1
            filename += str(num)

            with open(filename, 'w') as json_file: json.dump(self.data, json_file, indent=4) # save history

        self.changed = False
        return self


    @staticmethod
    def load(video_path: str, funscript_path: str):
        """ Load funscript from file

        Args:
            funscript_path (path): funscript path
            video_path (path): video path

        Returns:
            Funscript: a new funscript object
        """
        with open(funscript_path, 'r') as json_file:
            data = json.loads(json_file.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        funscript = Funscript(fps=fps)
        funscript.data = data

        if 'inverted' in funscript.data.keys() and funscript.data['inverted'] == True:
            funscript.invert_actions()
            funscript.data['inverted'] = False

        if 'fps' not in funscript.data.keys():
            funscript.data['fps'] = fps

        return funscript, funscript_path
