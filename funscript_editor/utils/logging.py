""" Methods to setup the logging """

import os
import yaml
import platform
import logging
import coloredlogs
import logging.config

from funscript_editor.definitions import WINDOWS_LOG_CONFIG_FILE, LINUX_LOG_CONFIG_FILE

def create_log_directories(config: dict) -> None:
    """ create all log directories for a log configuration

    Args:
        config (dict): the logging configuration dictionary
    """
    if isinstance(config, dict):
        for k in config.keys():
            create_log_directories(config[k])
            if k == 'filename':
                os.makedirs(os.path.dirname(os.path.abspath(config[k])), exist_ok=True)


def get_log_config_path() -> str:
    """ Get the log config file path for current platfrom

    Returns:
        str: the log config file path
    """
    return WINDOWS_LOG_CONFIG_FILE if platform.system() == 'Windows' else LINUX_LOG_CONFIG_FILE


class DevZeroLogger:
    """ Logger replacement to suppresses all log messages

    Args:
        name (str): name of the logger instance
    """

    def __init__(self, name):
        self.name = name

    def debug(self, *args):
        pass

    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def error(self, *args):
        pass

    def critical(self, *args):
        pass


def getLogger(name) -> DevZeroLogger:
    """ Get logger wrapper for python logging.getLogger

    Args:
        name (str): name of the logger instance
    """
    return DevZeroLogger(name)


def get_logfiles_paths() -> list:
    """ Get the logfiles paths from log config

    Returns:
        list: all logiles paths
    """
    try:
        result = []
        config_path = get_log_config_path()
        with open(config_path, 'rt') as f:
            for line in f.readlines():
                if "filename:" in line:
                    result.append(line.split(':')[1].strip())

        return result
    except:
        return []


def setup_logging(
        default_level :int = logging.INFO,
        env_key :str = 'LOG_CFG') -> None:
    """ Logging Setup

    Args:
        default_level (int): logging level e.g. `logging.INFO` (default is `logging.DEBUG`).
        env_key (str, optional): env variable name to load a configuration file via environment variable (default is `LOG_CFG`).
    """
    config_path = get_log_config_path()
    value = os.getenv(env_key, None)
    if value: config_path = value
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                create_log_directories(config)
                logging.config.dictConfig(config)
                coloredlogs.install(level=default_level)
                logging.debug('Loging setup completed')
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Failed to load configuration file. Using default configs')
