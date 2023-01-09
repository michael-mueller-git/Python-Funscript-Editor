@echo off

rmdir /Q /S "build" 2>NUL
rmdir /Q /S "dist/funscript-editor" 2>NUL
del "dist\\funscript-editor.zip" 2>NUL
del funscript-editor.spec 2>NUL

set CONDAPATH=%USERPROFILE%\Miniconda3
set ENVNAME=python-funscript-editor

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

call %CONDAPATH%\Scripts\activate.bat base

if exist %ENVPATH% (
  call conda env update --name %ENVNAME% --file environment_windows.yaml --prune
) else (
  call conda env create -f environment_windows.yaml --name %ENVNAME%
)

call conda activate %ENVNAME%

cd docs/app
mkdocs build
cd ../..
pyinstaller --add-data="funscript_editor/config/*;funscript_editor/config/" --add-data="assets/*;./" --hidden-import "pynput.keyboard._win32" --hidden-import "pynput.mouse._win32" --noupx --icon=icon.ico funscript-editor.py
move "docs\\app\\site" "dist\\funscript-editor\\funscript_editor\\docs"
copy /Y "funscript_editor\\VERSION.txt" "dist\\funscript-editor\\funscript_editor"
md "dist\\funscript-editor\\OFS"
copy /Y "contrib\\Installer\\assets\\main.lua" "dist\\funscript-editor\\OFS"
copy /Y "contrib\\Installer\\assets\\json.lua" "dist\\funscript-editor\\OFS"
powershell Compress-Archive -LiteralPath "dist/funscript-editor" -DestinationPath "dist/funscript-editor.zip"

call conda deactivate
timeout 20 >NUL
