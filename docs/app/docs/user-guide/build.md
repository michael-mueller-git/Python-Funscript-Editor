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

Ubuntu use the GTK Backend for the preview window. This will cause freezes of the UI, because i use Qt threads in my code. The Arch Linux OpenCV library was compiled with QT (see `python3 -c "import cv2; print(cv2.getBuildInformation())"` output) so no problem here. To use the Application on Ubuntu you have to compile `OpenCV + OpenCV contrib` with `-D WITH_QT=ON` flag from source. Or simply use [miniconda](https://docs.conda.io/en/latest/miniconda.html). They include OpenCV compiled with Qt support. After you have setup miniconda simply type the following commands to use Python-Funscript-Editor on Ubuntu:

```bash
sudo apt install -y make git gcc g++ libmpv-dev
sudo add-apt-repository ppa:savoury1/ffmpeg4
sudo apt install -y ffmpeg
git clone https://github.com/michael-mueller-git/Python-Funscript-Editor.git
conda env create -f environment_ubuntu.yaml
conda activate funscript-editor
make docs
python3 main.py
```

**NOTE:** The ffmpeg in the Ubuntu 20.04 LTS ppa package is too old, which means that the package does not contain the `v360` filter for VR videos. Therefore we install ffmpeg from savoury1 ppa.
