# Python Funscript Editor Config

The configuration files for the Windows release version are located in the `funscript-editor/funscript_editor/config` directory since release `v0.0.3`. If you use the python source code directly they are located in [`funscript_editor/config`](https://github.com/michael-mueller-git/Python-Funscript-Editor/tree/main/funscript_editor/config).

## Config Files

The directory contains several config files. The most interesting are `video_scaling.json`, `settings.yaml` and `hyperparameter.yaml`. When editing the `*.yaml` configuration files, pay attention to the formatting, otherwise the program will not work later.

Config Files:

- `hyperparameter.yaml`: hyperparameter for the prediction algorithms
- `logging_linux.yaml`: the logging configuration for linx
- `logging_windows.yaml`: the logging configuration for windows
- `settings.yaml`: application settings
- `ui.yaml`: user interface settings
- `video_scaling.json`: scaling for the preview window

### Config Parameter

#### `hyperparameter.yaml`

- `skip_frames` (int): This parameter specifies how many frames are skipped and interpolated during tracking. Increase this parameter to improve the processing speed on slow hardware. But higher values result in poorer predictions!
- `avg_sec_for_local_min_max_extraction` (float): Specify the window size for the calculation of the reference value for the local min and max search.
- `min_frames` (int): Specify the minimum required frames for the tracking. Wee need this parameter to ensure there is at leas two strokes in the tracking result.
- `shift_top_points` (int): Shift predicted top points by given frame number. Positive values delay the position and negative values result in an earlier position.
- `shift_bottom_points` (int): Shift predicted bottom points by given frame number. Positive values delay the position and negative values result in an earlier position.

#### `settings.yaml`

- `use_zoom` (bool): Enable or disable an additional step to zoom in the Video before selecting a tracking feature for the Woman or Men.
- `tracking_direction` (str): Specify the tracking direction. Allowed values are `'x'` and `'y'`.
- `max_playback_fps` (int): Limit the max player speed in the tracking preview window (0 = disable limit)

#### `video_scaling.json`

The `video_scaling.json` config file specifies how the video get scaled bevor the tracking. The scaling also apply tho the preview size. If the preview to select the tracking feature is to small or to large you have to adjust this config file.

The entries in this config file consist of a pair of values.

Example config:

```json
{ "1920": 1.0, "3500": 0.5, "5000": 0.25 }
```

One pair in the example config is e.g. `"1920": 1.0`. Each pair of values defines which scaling should be used for which video resolution. The first value, refers to the video width in pixels. Videos with size larger than 1920 pixels horizontally use a scaling of `1.0`. Videos with 3500 pixel and more are scaled with `0.5` and from 5000 with `0.25`. All videos which are smaller than the smallest value (in this case 1920) are scaled always with `1.0` (original size). You can enter as many values as you want and change the existing scaling.

It’s best to look at your screen resolution and calculate which scaling you need for which video size so that the window fits on the monitor. e.g. You have `1920x1080` screen and `5400x2700` Video, you can divide `1920 / 5400 = 0.36` → add `"5300": 0.35` to the config (the key value have to be a little bit smaller than the Video resolution to apply the correct scaling).
