#!/bin/bash

sudo sysctl kernel.unprivileged_userns_clone=1
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install -y flathub org.freedesktop.Platform//21.08 org.freedesktop.Sdk//21.08
flatpak install -y flathub org.gnome.Platform//3.38 org.gnome.Sdk//3.38
flatpak install -y flathub org.gtk.Gtk3theme.Breeze-Dark
flatpak-builder --repo=repo --force-clean build-dir org.flatpak.PythonFunscriptEditor.json
