# Open Funscripter Integration

Currently we use a hacky lua script to communicate between the Python Funscript Generator and the Open Funscripter.

## Installation (Windows)

1. Download the **latest** packed Python Funscript Editor from [github release page](https://github.com/michael-mueller-git/Python-Funscript-Editor/releases).
2. Extract the Archiv.
3. Download the **latest** `funscript_generator.lua` script from [github release page](https://github.com/michael-mueller-git/Python-Funscript-Editor/releases).
4. Move the downloaded `funscript_generator.lua` script to `data/lua` in your OFS directory.
5. Open the `funscript_generator.lua` file and adjust the `Settings.FunscriptGenerator` and `Settings.TmpFile` variable.
   - **NOTE:** You have to use use `/` or `\\` for the `\` symbols in your path!
   - `Settings.FunscriptGenerator`: Point to the extracted Python Funscript Editor program (better double check the complete path string to avoid errors later on).
   - `Settings.TmpFile`: Specifies a temporary file location where to store the result (must be a file not a directory!). The file does not have to exist at the moment. The specified file will be generated from the Python Funscript Editor and will be overwritten automatically at the next time the generator is started!
6. Now launch OFS.
7. Navigate to `View : Special functions : Custom Functions` and select the `funscript_generator.lua` entry. Click the Button `Bind Script` (This may trigger the funscript generator, just ignore it for now).
8. Navigate to `Options : Keys : Dynamic` and insert a shortcut for the funscript generator.
9. Now you can use the shortcut at any position in the video to start the funscript generator.

## Troubleshot

### After setting the min and max value after tracking no points are inserted in OFS

If the points are missing in OFS then most likely the variable `Settings.TmpFile` in the `funscript_generator.lua` script is set incorrectly.

### Noting happens when i press the Shortcut for the Funscript Generator

In most cases, the variable `Settings.FunscriptGenerator` in the `funscript_generator.lua` script was not set correctly.

If you are using an older version of the `funscript_generator.lua` you should download the latest version of the script from the [github release page](https://github.com/michael-mueller-git/Python-Funscript-Editor/releases). With older `funscript_generator.lua` scripts there were e.g. problems with blanks in the path.

### When calling the generator, only a message box is displayed with the message: "Video file was not specified!"

In some cases OFS does not set the path to the video file within the lua script correctly (the variable `VideoFilePath` is empty). Mostly it helps to save the current project and/or create a new project. Then the variable should be set by OFS and the generator should work.

### Tracking stops automatically

If a tacker does not find the selected feature in the next frame, the tracing process stops. If more than 120 frames have already been tracked, a window appears to select the minimum and maximum value in which the reason for the abort is displayed with e.g. `Info: Tracker Woman Lost`.

If less than 120 frames have been processed, a message box should pop up with the message `Tracking time insufficient`. In this case, no output is generated because there is not enough data for the algorithm to work with.
