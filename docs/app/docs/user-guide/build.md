# Build

## Windows

Use `pyinstaller` in anaconda environment with all packages from `requirements.txt` set up. Then run:

```
pip install pyinstaller
build.bat
```

This create the Windows Package in `./dist`

## Pip-Package (Recommend for Linux)

Generate distribution package of this project. These are archives that can be uploaded to an local Package Index and can be installed by pip.

```bash
make docs package
```

This create the distribution package in `./dist`. Or simply type `make all` to build and install the package.
