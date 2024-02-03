{
  description = "MTFG";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.05";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = "${system}";
      };
      mtfgDependencies = with pkgs; [
        qt5.qtbase
        qt5.full
        qt5.wrapQtAppsHook
        libsForQt5.breeze-qt5
        libsForQt5.qt5ct
        customPythonPackages.pkgs.opencv4
        customPythonPackages.pkgs.simplification
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
      libPath = pkgs.lib.makeLibraryPath mtfgDependencies;
      binPath = pkgs.lib.makeBinPath mtfgDependencies;

      customPythonPackages =
        let
          packageOverrides = self:
            super: {
              opencv4 = super.opencv4.overrideAttrs (old: rec {
                buildInputs = old.buildInputs ++ [ pkgs.qt5.full ];
                cmakeFlags = old.cmakeFlags ++ [ "-DWITH_QT=ON" ];
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
        pkgs.python39.override { inherit packageOverrides; self = customPythonPackages; };
    in
    {
      packages.${system}.mtfg = pkgs.stdenv.mkDerivation {
        pname = "MTFG";
        version = "0.5.3";
        src = pkgs.fetchgit {
          url = "https://github.com/michael-mueller-git/Python-Funscript-Editor.git";
          rev = "7cb898de9fec72c7e7b0ce2c3f25ad52f1d3a24e";
          sha256 = "sha256-SMHaGIMkoVdc2gENR1ILV/H+BZ7yHB4yxcpVxJaY9oo=";
        };
        buildInputs = mtfgDependencies;
        QT_QPA_PLATFORM = "xcb";
      };
      defaultPackage.${system} = self.packages.x86_64-linux.mtfg;
      formatter.${system} = pkgs.nixpkgs-fmt;
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = mtfgDependencies;
        shellHook = ''
          # use xwayland not wayland
          export QT_QPA_PLATFORM="xcb"
        '';
      };
    };
}
