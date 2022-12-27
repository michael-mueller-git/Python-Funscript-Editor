# usage: `nix develop --impure`
{
  description = "MTFG";

  inputs = {
    nixpkgs = {
        url = "github:NixOS/nixpkgs/nixos-22.05";
      };
      flake-utils = {
        url = "github:numtide/flake-utils";
      };
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in
        {
          devShells.default = import ./shell.nix { inherit pkgs; };
        }
      );
}
