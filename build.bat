@echo off
rmdir /Q /S "build" 2>NUL
rmdir /Q /S "dist/funscript-editor" 2>NUL
del "dist\\funscript-editor.zip" 2>NUL
del funscript-editor.spec 2>NUL
cd docs/app
mkdocs build
cd ../..
pyinstaller --add-data="funscript_editor/config/*;funscript_editor/config/" --add-data="assets/*;./" --hidden-import "pynput.keyboard._win32" --hidden-import "pynput.mouse._win32" --noupx --noconsole --icon=icon.ico funscript-editor.py
move "docs\\app\\site" "dist\\funscript-editor\\funscript_editor\\docs"
copy /Y "funscript_editor\\VERSION.txt" "dist\\funscript-editor\\funscript_editor"
powershell Compress-Archive -LiteralPath "dist/funscript-editor" -DestinationPath "dist/funscript-editor.zip"

