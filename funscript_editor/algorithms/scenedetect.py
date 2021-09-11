import csv
import os
import logging


class SceneDetectFromFile:
    """ Read the Scenes.csv file from [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).

    Note:
        Use "scenedetect -i video.mkv detect-content list-scenes" command to generate the required Scenes.csv file.


    Args:
        video_path (str): path to current video file
    """

    def __init__(self,
            video_path: str):
        self.logger = logging.getLogger(__name__)
        self.__set_csv_path(video_path)
        self.__load_scenes()


    def __set_csv_path(self, video_path: str) -> None:
        self.scenes_csv_path = str('.'.join(video_path.split('.')[0:-1]) + '-Scenes.csv')


    def __load_scenes(self) -> None:
        self.scenes = []
        if not os.path.exists(self.scenes_csv_path):
            self.logger.info("Scenes CSV File not found")
        else:
            try:
                with open(self.scenes_csv_path) as csvDataFile:
                    csvReader = csv.reader(csvDataFile)
                    found_header = False
                    for row in csvReader:
                        if found_header:
                            # Column 2 (index 1) is Start Frame number
                            try: self.scenes.append(int(row[1]))
                            except: pass
                        elif row[1] == 'Start Frame':
                            found_header = True

                    self.logger.info("Load %d scenes from %s", len(self.scenes), self.scenes_csv_path)
            except:
                self.logger.warning("Scenes CSV has wrong format!")


    def is_scene_change(self, frame_number: int) -> bool:
        """ Check if current frame number is assosiated with an scene change

        Args:
            frame_number (int): current frame number

        Returns:
            bool: True if current frame belongs to an scene change else False

        """
        if frame_number in self.scenes:
            self.logger.info("Detect scene change")
            return True

        return False
