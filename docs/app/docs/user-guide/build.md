# Build from Source

For Windows user i recommend to use the release version from [github release page](https://github.com/michael-mueller-git/Python-Funscript-Editor/releases)

## Windows 10

We use [pyinstaller](https://pypi.org/project/pyinstaller/) in [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment to create an Windows executable.

First download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (If you have already [anaconda](https://www.anaconda.com/) installed on your computer you can skip this step). Then run the following commands in the project root directory:

```
conda env create -f environment_windows.yaml
conda activate build
build.bat
```

This create the Windows executable in `./dist`.

Finally you can remove the build environment with `conda env remove -n build`

## Ubuntu 20.04 LTS, Ubuntu 21.04, Debian

The OpenCV Package in Ubuntu use the GTK Backend for the preview window. This will cause freezes of the UI, because i use Qt threads in my code. The Arch Linux OpenCV library was compiled with QT (see `python3 -c "import cv2; print(cv2.getBuildInformation())"` output) so no problem here. To use the Application on Ubuntu you have to compile `OpenCV + OpenCV contrib` with `-D WITH_QT=ON` flag from source. Or simply use [miniconda](https://docs.conda.io/en/latest/miniconda.html). They include OpenCV compiled with Qt support.

**NOTE:** I have also test this setup on Ubuntu 21.04 with Wayland desktop. You can use the setup instructions from Ubuntu 20.04 LTS. The application work with XWayland on the Wayland desktop!

### Miniconda

After you have setup [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your machine simply type the following commands to use Python-Funscript-Editor:

```bash
sudo apt install -y make git gcc g++ libmpv-dev libatlas-base-dev
git clone https://github.com/michael-mueller-git/Python-Funscript-Editor.git
cd Python-Funscript-Editor
conda env create -f environment_ubuntu.yaml
```

On some distributions e.g. Ubuntu 20.04 LTS, the FFmpeg package does not contain the required `v360` filter. Therefore you can download a static linked FFmpeg package for this application with `bash download_ffmpeg.sh` from project root directory. (Perform this step if you are not sure if your FFmpeg version is sufficient!).

Ubuntu user can alternatively use the `savoury1` ppa:

```bash
sudo add-apt-repository ppa:savoury1/ffmpeg4
sudo apt install -y ffmpeg

```

Now you can run the program direct from source code with:

```bash
conda activate funscript-editor
python3 funscript-editor.py
```

By default the latest code is used which may contain bugs. So you maybe want to switch to the latest release version with `` git checkout $(git describe --tags `git rev-list --tags --max-count=1`)  ``.

You can always update the application to the latest version with `git checkout main && git pull` inside the repository directory.
