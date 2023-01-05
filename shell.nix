{ pkgs ? import <nixpkgs> {} }:
let python =
    let
    packageOverrides = self:
    super: {
      opencv4 = super.opencv4.overrideAttrs (old: rec {
        buildInputs = old.buildInputs ++ [pkgs.qt5.full];
        cmakeFlags = old.cmakeFlags ++ ["-DWITH_QT=ON"];
      });
    };
    in
      pkgs.python39.override {inherit packageOverrides; self = python;};
in
  pkgs.mkShell {
    nativeBuildInputs = with pkgs; [
      qt5.qtbase
      qt5.full
      qt5.wrapQtAppsHook
      python.pkgs.opencv4
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
