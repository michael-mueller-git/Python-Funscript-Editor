# Python Funscript Editor

A Python program to quickly create prototype algorithms to partially automate the generation of funscripts.

**NOTE:** The program is currently not intended for productive use. But you can do whatever you want with this tool.

## Documentation

The application documentation is located in [`./docs/app/docs/index.md`](https://github.com/michael-mueller-git/Python-Funscript-Editor/blob/main/docs/app/docs/index.md)

## Build

### Windows

Use `pyinstaller` in anaconda environment with all packages from `requirements.txt` set up. Then run:

```
pip install pyinstaller
build.bat
```

NOTE: don't use `--onefile` with `pyinstaller`, this is way to slow

### Pip-Package (Recommend for Linux)

Generate distribution package of this project. These are archives that can be uploaded to an local Package Index and can be installed by pip.

```bash
make docs package
```

This create the distribution package in `./dist`. Or simply type `make all` to build and install the package.

## TODOs

- Append script to predict x-Movement
