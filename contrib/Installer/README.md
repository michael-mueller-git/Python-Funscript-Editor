# MTFG OFS Extension Installer

## Build Installer from Source

I recommend to download the installer from [release page](https://github.com/michael-mueller-git/Python-Funscript-Editor/releases) in the assets section.

If you want to build this installer from source use miniconda environment to build the executable.

First download and install miniconda (If you have already anaconda installed on your computer you can skip this step). Then run the following commands in the project root directory:

```
conda env create -f environment.yaml
conda activate build-installer
build.bat
```

This create the Windows executable in `./dist`.

Finally you can remove the build environment with `conda env remove -n build-installer`
