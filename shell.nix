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
      qt5.full
      python.pkgs.opencv4
      (python39.withPackages (p: with p; [
        coloredlogs
        cryptography
        matplotlib
        pillow
        playsound
        pynput
        pyqt5
        pip
        pyyaml
        scipy
        screeninfo
        mpv
      ]))
    ];

}
