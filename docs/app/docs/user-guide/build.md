# Build Application from Source

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
sudo apt install -y make git gcc g++ cmake libmpv-dev libatlas-base-dev
git clone https://github.com/michael-mueller-git/Python-Funscript-Editor.git
cd Python-Funscript-Editor
conda env create -f environment_ubuntu.yaml
```

On some distributions e.g. Ubuntu 20.04 LTS, the FFmpeg package does not contain the required `v360` filter. Therefore you can download a static linked FFmpeg package for this application with `bash download_ffmpeg.sh` from project root directory. (Perform this step if you are not sure if your FFmpeg version is sufficient!). The latest ffmpeg downloaded via `download_ffmpeg.sh` break current MTFG. Therefore you should copy the ffmpeg executable from `./assets` to `./funscript_editor/data` for now.

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

By default the latest code is used which may contain bugs. So you maybe want to switch to the latest release version with `` git checkout $(git describe --tags `git rev-list --tags --max-count=1`)``.

You can always update the application to the latest version with `git checkout main && git pull` inside the repository directory.

## Arch Linux and Arch-Based Distributions

My development platform is Arch Linux. The code should run therefore without problems. When installing the dependencies, make sure you install the following packages from the repositories, as the wrapper will have problems if the versions differ from the pip packages and the local C, C++ libraries.

```bash
sudo pacman -Syu  # this ensure the new libraries will be compatible
sudo pacman -Sy python-opencv python-pyqt5
```

All other required python packages can be installed from pip.

The latest ffmpeg break current MTFG. Therefore you should copy the ffmpeg executable from `./assets` to `./funscript_editor/data` for now!!

## Flatpak

The repository contains an flatpak build recipe that can be used to build an flatpak app. You can build the flatpak with the `build_flatpak.sh` script in the project root directory. The build script generate the `PythonFunscriptEditor.flatpak` package that can be installed with `flatpak install --user PythonFunscriptEditor.flatpak` on the system.

Limitation of the flatpak application:

- No public release in a flatpak repository available. You have to build the flatpak local.
- Slow build process.
- Flatpak need non-privileged user namespaces enabled (`sysctl kernel.unprivileged_userns_clone=1'`).
- The System-Theme is not applied to the application. The problem is that my recipe use conda packages which isolates the python packages inside the flatpak sandbox and we have no access to the system theme. To fix this we have to build all python packages in the flatpak recipe from source and include the system theme inside the flatpak.

## OFS Extension on Linux

### OFS 1.x.x

Path for extensions is `~/.local/share/OFS/OFS_data/extensions`

### OFS 2.x.x

Path for extensions is `~/.local/share/OFS/OFS2_data/extensions`

### OFS 3.x.x

Path for extensions is `~/.local/share/OFS/OFS3_data/extensions`
