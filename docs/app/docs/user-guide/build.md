# Build from Source

For Windows user i recommend to use the release version from [github release page](https://github.com/michael-mueller-git/Python-Funscript-Editor/releases)

## Windows

We use [pyinstaller](https://pypi.org/project/pyinstaller/) in [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment to create an Windows executable.

First download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (If you have already [anaconda](https://www.anaconda.com/) installed on your computer you can skip this step). Then run the following commands in the project root directory:

```
conda env create -f environment_windows.yaml
conda activate build
build.bat
```

This create the Windows executable in `./dist`.

Finally you can remove the build environment with `conda env remove -n build`

## Ubuntu

```bash
sudo apt update
sudo add-apt-repository ppa:savoury1/ffmpeg4
sudo apt install git python3 python3-pip python3-sphinx sphinx-rtd-theme-common python3-opencv mkdocs python3-pyqt5 ffmpeg libmpv-dev
git clone https://github.com/michael-mueller-git/Python-Funscript-Editor.git
cd Python-Funscript-Editor
pip install -r requirements.txt
make docs
```

Now you can run the code direct with `python3 funscript-editor.py` or install install a pip package with `make all`.

**NOTE:** In my Ubuntu test setup the UI hangs because Ubuntu use the GTK Backend while the Arch Linux OpenCV library was compiled with QT (see `python3 -c "import cv2; print(cv2.getBuildInformation())"` output). To use the Application on Ubuntu you have to compile OpenCV + OpenCV contrib with `-D WITH_QT=ON` flag from source.
