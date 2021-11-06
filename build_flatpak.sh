#!/bin/bash

option="$1"
[ -z "$option" ] && option="--default"

sudo sysctl kernel.unprivileged_userns_clone=1
sudo flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
sudo flatpak install -y flathub org.kde.Platform//5.15 org.kde.Sdk//5.15

if [ "$option" = "--install" ]; then
    flatpak-builder --user --install --force-clean build-dir org.flatpak.PythonFunscriptEditor.json
else
    flatpak-builder --repo=repo --force-clean build-dir org.flatpak.PythonFunscriptEditor.json
    flatpak build-bundle repo PythonFunscriptEditor.flatpak org.flatpak.PythonFunscriptEditor
fi

sudo sysctl kernel.unprivileged_userns_clone=0
