{ pkgs ? import <nixpkgs> {} }:
let python =
    let
    packageOverrides = self:
    super: {
      opencv4 = super.opencv4.overrideAttrs (old: rec {
        buildInputs = old.buildInputs ++ [pkgs.qt5.full];
        cmakeFlags = old.cmakeFlags ++ ["-DWITH_QT=ON"];
      });

      simplification = pkgs.python39Packages.buildPythonPackage rec {
        pname = "simplification";
        version = "0.6.2";
        format = "wheel";

        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/02/3e/829b59a5d072feb45e14879d3149a2dad743a18f83db29d8f3800a33aa64/simplification-0.6.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
          sha256 = "20fb00f219bdd33319fc7526d23ef1fb7e52a40027a010f1013dd60626282325";
        };

        doCheck = false;
      };
    };
    in
      pkgs.python39.override {inherit packageOverrides; self = python;};
in
  pkgs.mkShell {
    nativeBuildInputs = with pkgs; [
      qt5.qtbase
      qt5.full
      qt5.wrapQtAppsHook
      libsForQt5.breeze-qt5
      libsForQt5.qt5ct
      python.pkgs.opencv4
      python.pkgs.simplification
      (python39.withPackages (p: with p; [
        coloredlogs
        cryptography
        matplotlib
        mpv
        pillow
        pip
        playsound
        pynput
        pyqt5
        pyqtgraph
        pyyaml
        scipy
        screeninfo
      ]))
    ];
}
