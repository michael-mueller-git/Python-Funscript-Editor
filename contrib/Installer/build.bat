@echo off
rmdir /Q /S "build" 2>NUL
rmdir /Q /S "dist" 2>NUL
del mtfg-ofs-extension-installer.spec 2>NUL
pyinstaller --add-data="assets/*;./" --noupx --onefile mtfg-ofs-extension-installer.py

