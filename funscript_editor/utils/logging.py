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

def setup_logging(
        default_level :int = logging.INFO,
        env_key :str = 'LOG_CFG') -> None:
    """ Logging Setup

    Args:
        default_level (int): logging level from logging python module e.g. `logging.INFO` (default is `logging.INFO`).
        env_key (str, optional): env variable name to load a configuration file via environment variable (default is `LOG_CFG`).
    """
    config_path = WINDOWS_LOG_CONFIG_FILE if platform.system() == 'Windows' else LINUX_LOG_CONFIG_FILE
    value = os.getenv(env_key, None)
    if value: config_path = value
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                create_log_directories(config)
                logging.config.dictConfig(config)
                coloredlogs.install()
            except Exception as e:
                print(e)
                print('Error in Logging Configuration. Using default configs')
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        print('Failed to load configuration file. Using default configs')
