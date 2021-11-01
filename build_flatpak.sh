#!/bin/bash

sudo sysctl kernel.unprivileged_userns_clone=1
sudo flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
sudo flatpak install -y flathub org.freedesktop.Platform//21.08 org.freedesktop.Sdk//21.08
sudo flatpak install -y flathub org.gnome.Platform//3.38 org.gnome.Sdk//3.38
sudo flatpak install -y flathub org.kde.Platform//5.15 org.kde.Sdk//5.15
sudo flatpak install -y flathub org.gtk.Gtk3theme.Breeze-Dark
flatpak-builder --repo=repo --force-clean build-dir org.flatpak.PythonFunscriptEditor.json
