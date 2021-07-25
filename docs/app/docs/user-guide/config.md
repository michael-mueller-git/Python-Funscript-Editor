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

### Config Parameter

#### `settings.yaml`

- `use_kalman_filter`: (bool) Enable Kalman Filter
- `use_zoom` (bool): Enable or disable an additional step to zoom in the Video before selecting a tracking feature for the Woman or Men.
- `zoom_factor:` (float): Set the desired zoom value which will be used when the zoom function is activated.
- `max_playback_fps` (int): Limit the max player speed in the tracking preview window (0 = disable limit)
- `preview_scaling` (float): Set the preview image scaling factor. With a value of `1.0`, the window should fill the height or width of the screen depending on the aspect ratio of the video.
- `tracker`: (str) Specify the tracker algorithm. Available options are `'MIL'`, `'KCF'`, `'CSRT'`.

#### `hyperparameter.yaml`

- `skip_frames` (int): This parameter specifies how many frames are skipped and interpolated during tracking. Increase this parameter to improve the processing speed on slow hardware. But higher values result in poorer predictions!
- `avg_sec_for_local_min_max_extraction` (float): Specify the window size for the calculation of the reference value for the local min and max search.
- `local_max_delta_in_percent` (float): Specify the maximum deviation for a max point in percent (recommend range: `0.0 - 10.0`)
- `local_min_delta_in_percent` (float): Specify the maximum deviation for a min point in percent (recommend range: `0.0 - 10.0`)
- `min_frames` (int): Specify the minimum required frames for the tracking. Wee need this parameter to ensure there is at leas two strokes in the tracking result.
- `shift_top_points` (int): Shift predicted top points by given frame number. Positive values delay the position and negative values result in an earlier position.
- `shift_bottom_points` (int): Shift predicted bottom points by given frame number. Positive values delay the position and negative values result in an earlier position.
- `top_points_offset` (float): An fix offset to the top points (positive values move the point up and negative values move the point down). The offset respect the user defined upper and lower limit.
- `bottom_points_offset` (float): An fix offset to the bottom points (positive values move the point up and negative values move the point down). The offset respect the user defined upper and lower limit.
- `top_threshold` (float): Define the top threshold. All top points greater than `(max - threshold)` will be set to the specified max value. Set 0.0 to disable this function.
- `bottom_threshold` (float): Define the bottom threshold. All bottom points lower than `(min + threshold)` will be set to the specified min value. Set 0.0 to disable this function.
