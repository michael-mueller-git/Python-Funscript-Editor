# Build Application from Source

For Windows user i recommend to use the release version from [github release page](https://github.com/michael-mueller-git/Python-Funscript-Editor/releases)

## Windows 10/11

We use [pyinstaller](https://pypi.org/project/pyinstaller/) in [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment to create an Windows executable.

First download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then double click the `build.bat` int the repository root directory to start the build process. This create the Windows executable in `./dist`.

## Linux

To run the application on Linux you need the [nix package manager](https://nixos.org/download.html) with experimental [`flakes` feature enabled](https://github.com/mschwaig/howto-install-nix-with-flake-support). Open an environment with `nix develop` in the repository root directory and start the application with `python main.py` from activated nix develop shell.

## Flatpak

The repository contains an flatpak build recipe that can be used to build an flatpak app. You can build the flatpak with the `build_flatpak.sh` script in the project root directory. The build script generate the `PythonFunscriptEditor.flatpak` package that can be installed with `flatpak install --user PythonFunscriptEditor.flatpak` on the system.

Limitation of the flatpak application:

- No public release in a flatpak repository available. You have to build the flatpak local.
- Slow build process.
- Flatpak need non-privileged user namespaces enabled (`sysctl kernel.unprivileged_userns_clone=1'`).
- The System-Theme is not applied to the application. The problem is that my recipe use conda packages which isolates the python packages inside the flatpak sandbox and we have no access to the system theme. To fix this we have to build all python packages in the flatpak recipe from source and include the system theme inside the flatpak.

## OFS Extension Path on Linux

Path for extensions is `~/.local/share/OFS/OFSX_data/extensions`
