# Flatpak

## Build Flatpak from Source

Run in project root:

```bash
./build_flatpak.sh
```

To verify that the build was successful, run the following:

```bash
flatpak --user remote-add --no-gpg-verify local-repo repo
flatpak --user install -y local-repo org.flatpak.PythonFunscriptEditor
flatpak run org.flatpak.PythonFunscriptEditor
flatpak --user remote-delete local-repo
```

An other option is to use `flatpak-builder` to install the application.

```
flatpak-builder --user --install --force-clean build-dir org.flatpak.PythonFunscriptEditor.json
flatpak run org.flatpak.PythonFunscriptEditor
flatpak uninstall org.flatpak.PythonFunscriptEditor
```

To install the `*.flatpak` bundle from build script you can use:

```bash
flatpak install --user PythonFunscriptEditor.flatpak
flatpak run org.flatpak.PythonFunscriptEditor
```
