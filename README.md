# Python Funscript Editor

A Python program to quickly create prototype algorithms to partially automate the generation of funscripts.

**NOTE:** The program is currently not intended for productive use. But you can do whatever you want with this tool.

## Documentation

- The application documentation is located in `./docs/app/site/` (`index.html`).
- The code documentation is located in `./docs/code/_build/html/` (`index.html`).

## Build

### Windows

Use `pyinstaller` in anaconda environment with all required packages set up:

```
pip install pyinstaller
pyinstaller --hidden-import "pynput.keyboard._win32" --hidden-import "pynput.mouse._win32" main.py
```

NOTE: don't use `--onefile`, this is way to slow

### Pip-Package (Recommend for Linux)

Generate distribution package of this project. These are archives that can be uploaded to an local Package Index and can be installed by pip.

```bash
make docs package
```

This create the distribution package in `./dist`.

## TODOs

- Append script to predict x-Movement
