{
  description = "python devshell with uv and cuda support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-python,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    {
      devShell.${system} = pkgs.mkShell {
        buildInputs = with pkgs; [
          nixpkgs-python.packages.${system}."3.13.2"
          pkgs.nodejs_20
          uv
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          ffmpeg
          libsndfile
          pkgs.stdenv.cc.cc.lib
        ];

        shellHook = ''
          export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [
              pkgs.zlib
              pkgs.stdenv.cc.cc
              pkgs.stdenv.cc.cc.lib
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.libsndfile
            ]
          }:/run/opengl-driver/lib:$LD_LIBRARY_PATH
          export UV_SYSTEM_PYTHON=0
          if [ -f "$PWD/.venv/bin/activate" ]; then
            source "$PWD/.venv/bin/activate"
          fi
        '';
      };
    };
}
