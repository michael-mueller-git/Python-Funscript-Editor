# Python Funscript Editor Config

The configuration files for the Windows release version are located in the `funscript-editor/funscript_editor/config` directory. If you use the python source code directly they are located in [`funscript_editor/config`](https://github.com/michael-mueller-git/Python-Funscript-Editor/tree/main/funscript_editor/config).

## Config Files

The directory contains several config files. The most interesting are `settings.yaml` and `hyperparameter.yaml`. When editing the `*.yaml` configuration files, pay attention to the formatting, otherwise the program will not work later.

Config Files:

- `hyperparameter.yaml`: hyperparameter for the algorithms
- `logging_linux.yaml`: the logging configuration for linux
- `logging_windows.yaml`: the logging configuration for windows
- `settings.yaml`: application settings
- `ui.yaml`: user interface settings
- `projection.yaml`: the FFmpeg filter parameter for different video types

All available config parameter are described with an comment inside the config file.
