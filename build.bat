@echo off
rmdir /Q /S "build" >/NUL
rmdir /Q /S "dist/funscript-editor" >NUL
del funscript-editor.spec >NUL
pyinstaller --add-data="funscript_editor/config/*;funscript_editor/config" --hidden-import "pynput.keyboard._win32" --hidden-import "pynput.mouse._win32" funscript-editor.py
xcopy /s "assets" "dist/funscript-editor"