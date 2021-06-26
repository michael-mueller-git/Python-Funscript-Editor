# Open Funscripter Integration

Currently we use a hacky lua script to communicate between the Python Funscript Generator and the Open Funscripter.

## Installation

1. Download the latest packed Python Funscript Editor from [github release page](https://github.com/michael-mueller-git/Python-Funscript-Editor/releases).
2. Extract the Archiv to an Path without special character or spaces.
3. Copy the `funscript_generator.lua` script ([`Repositor/contrib/OpenFunscripter`](https://github.com/michael-mueller-git/Python-Funscript-Editor/tree/main/contrib/OpenFunscripter)) to `data/lua` in your OFS directory.
4. Open the `funscript_generator.lua` file and adjust the `Settings.FunscriptGenerator` and `Settings.TmpFile` variable.
   - `Settings.FunscriptGenerator`: point to the extracted Python Funscript Editor program
   - `Settings.TmpFile`: specifies a temporary file where to store the result (must be a file not a directory!). The file will be overwritten automatically the next time the generator is started!
5. Now launch OFS.
6. Navigate to `View : Special functions : Custom Functions` and select the `funscript_generator.lua` entry. Click the Button `Bind Script` (This may trigger the funscript generator, just ignore it for now).
7. Navigate to `Options : Keys : Dynamic` and insert a shortcut for the funscript generator.
8. Now you can use the shortcut at any position in the video to start the funscript generator.
