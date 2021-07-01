""" Configuration parser """

import os
import yaml

from funscript_editor.definitions import ROOT_DIR, UI_CONFIG_FILE, \
        HYPERPARAMETER_CONFIG_FILE, SETTINGS_CONFIG_FILE, PROJECTION_CONFIG_FILE

def read_yaml_config(config_file: str) -> dict:
    """ Parse a yaml config file

    Args:
        config_file (str): path to config file to parse

    Returns:
        dict: the configuration dictionary
    """
    with open(config_file) as f:
        return yaml.load(f, Loader = yaml.FullLoader)

def read_version() -> str:
    """ Red current package version

    Returns:
        str: package version
    """
    version_file = os.path.join(ROOT_DIR, 'VERSION.txt')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            return f.read().replace('v', '').strip()
    return "0.0.0"

#: the package version
VERSION = read_version()

#: the ui config
UI_CONFIG = read_yaml_config(UI_CONFIG_FILE)

#: hyperparameter for the algorithms
HYPERPARAMETER = read_yaml_config(HYPERPARAMETER_CONFIG_FILE)

#: application settings
SETTINGS = read_yaml_config(SETTINGS_CONFIG_FILE)

#: projection parameter
PROJECTION = read_yaml_config(PROJECTION_CONFIG_FILE)
