@echo off
rmdir /Q /S "build" 2>NUL
rmdir /Q /S "dist/funscript-editor" 2>NUL
del funscript-editor.spec 2>NUL
pyinstaller --add-data="funscript_editor/config/*;funscript_editor/config" --hidden-import "pynput.keyboard._win32" --hidden-import "pynput.mouse._win32" funscript-editor.py
xcopy /s "assets" "dist/funscript-editor"
copy /Y "funscript_editor\\VERSION.txt" "dist\\funscript-editor\\funscript_editor"

