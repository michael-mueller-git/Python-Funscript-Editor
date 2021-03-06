import os
import sys
import platform

from pathlib import Path

import multiprocessing as mp


###############
# DIRECTORIES #
###############

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
UI_DIR = os.path.join(ROOT_DIR, 'ui')
ICON_PATH = os.path.join(CONFIG_DIR, 'icon.png')
DOCS_URL = "https://github.com/michael-mueller-git/Python-Funscript-Editor/tree/{tag}/docs/app/docs"

if os.path.exists(os.path.join(ROOT_DIR, '..', 'docs')):
    APP_DOCUMENTATION_DIR = os.path.join(ROOT_DIR, '..', 'docs', 'app', 'site')
    CODE_DOCUMENTATION_DIR = os.path.join(ROOT_DIR, '..', 'docs', 'code', '_build', 'html')
elif os.path.exists(os.path.join(sys.prefix, 'share', os.path.basename(ROOT_DIR), 'docs')):
    APP_DOCUMENTATION_DIR = os.path.join(sys.prefix, 'share', os.path.basename(ROOT_DIR), 'docs', 'app', 'site')
    CODE_DOCUMENTATION_DIR = os.path.join(sys.prefix, 'share', os.path.basename(ROOT_DIR), 'docs', 'code', '_build', 'html')
elif os.path.exists(os.path.join(ROOT_DIR, '..', 'usr', 'share', os.path.basename(ROOT_DIR), 'docs')):
    APP_DOCUMENTATION_DIR = os.path.join(ROOT_DIR, '..', 'usr', 'share', os.path.basename(ROOT_DIR), 'docs', 'app', 'site')
    CODE_DOCUMENTATION_DIR = os.path.join(ROOT_DIR, '..', 'usr', 'share', os.path.basename(ROOT_DIR), 'docs', 'code', '_build', 'html')
elif os.path.exists(os.path.join(ROOT_DIR, 'docs', 'app', 'site')):
    APP_DOCUMENTATION_DIR = os.path.join(ROOT_DIR, 'docs', 'app', 'site')
    CODE_DOCUMENTATION_DIR = os.path.join(ROOT_DIR, 'docs', 'code', '_build', 'html')
elif os.path.exists(os.path.join(ROOT_DIR, 'docs')):
    APP_DOCUMENTATION_DIR = os.path.join(ROOT_DIR, 'docs')
    CODE_DOCUMENTATION_DIR = '' # code documentation not included
else:
    APP_DOCUMENTATION_DIR = ''
    CODE_DOCUMENTATION_DIR = ''


################
# CONFIG FILES #
################

UI_CONFIG_FILE = os.path.join(CONFIG_DIR, 'ui.yaml')
WINDOWS_LOG_CONFIG_FILE = os.path.join(CONFIG_DIR, 'logging_windows.yaml')
LINUX_LOG_CONFIG_FILE = os.path.join(CONFIG_DIR, 'logging_linux.yaml')
HYPERPARAMETER_CONFIG_FILE = os.path.join(CONFIG_DIR, 'hyperparameter.yaml')
SETTINGS_CONFIG_FILE = os.path.join(CONFIG_DIR, 'settings.yaml')
PROJECTION_CONFIG_FILE = os.path.join(CONFIG_DIR, 'projection.yaml')


##########
# CONFIG #
##########

CPU_CORES = int(mp.cpu_count()-1)

