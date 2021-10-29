# MTFG OFS Extension Installer

**I have not released this install yet!**

## Usage

1. Download OFS from https://github.com/OpenFunscripter/OFS/releases.
2. Install OFS if you prefer the installer version (`exe`) else extract the OFS archive.
3. Start OFS at least one and close it again.
4. Download the MTFG OFS Extension Installer from https://github.com/michael-mueller-git/Python-Funscript-Editor/releases.
5. Execute the downloaded executable. Wait for the installation to complete.
6. Open OFS, activate the extension and enable the window.
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
