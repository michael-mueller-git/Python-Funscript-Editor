import os
import sys
import platform

from pathlib import Path

import multiprocessing as mp

CONFIG_VERSION = 1
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_TEMPLATE_DIR = os.path.join(ROOT_DIR, 'config')

if os.path.abspath(__file__).startswith("/nix"):
    CACHE_DIR = '/tmp/mtfg-cache'
    CONFIG_DIR = os.path.join(os.path.join(os.path.expanduser('~'), '.config'), 'mtfg')
    os.makedirs(CONFIG_DIR, exist_ok=True)
    COPY_CONFIG_TEMPLATES = True
    CONFIG_VERSION_FILE = os.path.join(CONFIG_DIR, ".version")
    if os.path.exists(CONFIG_VERSION_FILE):
        with open(CONFIG_VERSION_FILE, "r") as f:
            if f.readlines() == [str(CONFIG_VERSION)]:
                COPY_CONFIG_TEMPLATES = False
    if COPY_CONFIG_TEMPLATES:
        os.system(f"cp -rfv \"{CONFIG_TEMPLATE_DIR}/.\" \"{CONFIG_DIR}\"")
        with open(CONFIG_VERSION_FILE, "w") as f:
            f.write(str(CONFIG_VERSION))
        os.system(f"chmod 755 -R {CONFIG_DIR}")
else:
    CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
    CONFIG_DIR = CONFIG_TEMPLATE_DIR

ICON_PATH = os.path.join(os.path.join(ROOT_DIR, 'config'), 'icon.png')
DOCS_URL = "https://github.com/michael-mueller-git/Python-Funscript-Editor/tree/{tag}/docs/app/docs"
RAW_TRACKING_DATA_CAHCE_FILE = os.path.join(CACHE_DIR, 'raw_tracking_data.json')

os.makedirs(CACHE_DIR, exist_ok=True)

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

UI_CONFIG_FILE = os.path.join(CONFIG_DIR, 'ui.yaml')
WINDOWS_LOG_CONFIG_FILE = os.path.join(CONFIG_DIR, 'logging_windows.yaml')
LINUX_LOG_CONFIG_FILE = os.path.join(CONFIG_DIR, 'logging_linux.yaml')
HYPERPARAMETER_CONFIG_FILE = os.path.join(CONFIG_DIR, 'hyperparameter.yaml')
SETTINGS_CONFIG_FILE = os.path.join(CONFIG_DIR, 'settings.yaml')
PROJECTION_CONFIG_FILE = os.path.join(CONFIG_DIR, 'projection.yaml')
CPU_CORES = int(mp.cpu_count()-1)

