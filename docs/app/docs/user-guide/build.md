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

**NOTE:** In my Ubuntu test setup the UI hangs at all `cv2.waitKey()` lines in the code. Remove this part of the code solve the issue but the then i could not select an ROI, because the mouse event do not work. When i hard-code a trackingbox, the ui work but this is not usable. So we need future investigation for this problem. For now i recommend an Arch based Linux.
