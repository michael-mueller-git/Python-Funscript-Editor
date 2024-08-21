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
        ffmpeg_5
        qt5.qtbase
        qt5.full
        qt5.wrapQtAppsHook
        libsForQt5.breeze-qt5
        libsForQt5.qt5ct
        (python39.withPackages (p: with p; [
          customPythonPackages.pkgs.opencv4
          customPythonPackages.pkgs.simplification
          customPythonPackages.pkgs.coloredlogs
          customPythonPackages.pkgs.cryptography
          customPythonPackages.pkgs.matplotlib
          customPythonPackages.pkgs.mpv
          customPythonPackages.pkgs.pillow
          customPythonPackages.pkgs.pip
          customPythonPackages.pkgs.playsound
          customPythonPackages.pkgs.pynput
          customPythonPackages.pkgs.pyqt5
          customPythonPackages.pkgs.pyqtgraph
          customPythonPackages.pkgs.pyyaml
          customPythonPackages.pkgs.scipy
          customPythonPackages.pkgs.screeninfo
          customPythonPackages.pkgs.GitPython
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
      packages.${system}.mtfg = pkgs.python39Packages.buildPythonPackage {
        pname = "funscript-editor";
        version = "0.5.4";
        src = pkgs.fetchgit {
          url = "https://github.com/michael-mueller-git/Python-Funscript-Editor.git";
          rev = "eee779a3492397ade4d53129b9b42f6d50e83ec";
          sha256 = "sha256-5/2TEHQ/pIuRQul41oxjRmxwgt2c22/uoNvOnZXkj+I=";
        };
        propagatedBuildInputs = mtfgDependencies;
        nativeBuildInputs = with pkgs; [
          makeWrapper
        ];
        postInstall = ''
          wrapProgram "$out/bin/funscript-editor" --prefix LD_LIBRARY_PATH : "${libPath}" --prefix PATH : "${binPath}"
        '';
      };
      defaultPackage.${system} = self.packages.x86_64-linux.mtfg;
      formatter.${system} = pkgs.nixpkgs-fmt;
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = mtfgDependencies;
      };
    };
}
