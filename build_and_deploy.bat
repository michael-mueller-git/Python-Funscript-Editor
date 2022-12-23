@echo off
REM script to build an deploy ofs extension
rmdir /Q /S "build" 2>NUL
rmdir /Q /S "dist/funscript-editor" 2>NUL
rmdir /Q /S "%APPDATA%\\OFS\\OFS_data3\\extensions\\Funscript Generator Windows\\funscript-editor" 2>NUL
del "dist\\funscript-editor.zip" 2>NUL
del funscript-editor.spec 2>NUL
del "%APPDATA%\\OFS\\OFS_data3\\extensions\\Funscript Generator Windows\\main.lua" 2>NUL
del "%APPDATA%\\OFS\\OFS_data3\\extensions\\Funscript Generator Windows\\json.lua" 2>NUL
if not exist "%APPDATA%\\OFS\\OFS_data3\\extensions\\Funscript Generator Windows" mkdir "%APPDATA%\\OFS\\OFS_data3\\extensions\\Funscript Generator Windows"
cd docs/app
mkdocs build
cd ../..
pyinstaller --add-data="funscript_editor/config/*;funscript_editor/config/" --add-data="assets/*;./" --hidden-import "pynput.keyboard._win32" --hidden-import "pynput.mouse._win32" --noupx --icon=icon.ico funscript-editor.py
move "docs\\app\\site" "dist\\funscript-editor\\funscript_editor\\docs"
copy /Y "funscript_editor\\VERSION.txt" "dist\\funscript-editor\\funscript_editor"
md "dist\\funscript-editor\\OFS"
copy /Y "contrib\\Installer\\assets\\main.lua" "dist\\funscript-editor\\OFS"
copy /Y "contrib\\Installer\\assets\\json.lua" "dist\\funscript-editor\\OFS"
move "dist\\funscript-editor" "%APPDATA%\\OFS\\OFS_data3\\extensions\\Funscript Generator Windows\\funscript-editor"
copy /Y "contrib\\Installer\\assets\\main.lua" "%APPDATA%\\OFS\\OFS_data3\\extensions\\Funscript Generator Windows"
copy /Y "contrib\\Installer\\assets\\json.lua" "%APPDATA%\\OFS\\OFS_data3\\extensions\\Funscript Generator Windows"
