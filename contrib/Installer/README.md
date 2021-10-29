# MTFG OFS Extension Installer

**I have not released this install yet!**

## Usage

1. Download OFS from https://github.com/OpenFunscripter/OFS/releases.
   <br> ![OFS Download](./docs/ofs_installation_01.jpg)
2. Install OFS
3. Start OFS at least one and close it again.
4. Download the MTFG OFS Extension Installer from https://github.com/michael-mueller-git/Python-Funscript-Editor/releases.
5. Execute the downloaded executable. Wait for the installation to complete.
6. Open OFS, activate the extension (3) and enable the window (4). Now you can use the extension at any position in the Video with the _Start MTFG_ Button (5). On slow computers, the program may take several seconds to start!
   <br> ![Activate Extension](./docs/ofs_extension_02.jpg)
7. **Optional**: Add global key binding for the extension in OFS.
   <br> ![Assign an Shortcut](./docs/ofs_extension_03.jpg)

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
