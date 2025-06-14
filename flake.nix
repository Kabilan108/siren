{
  description = "python devshell with uv and cuda support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  outputs = { self, nixpkgs, nixpkgs-python }: let
    system = "x86_64-linux";
    pkgs   = import nixpkgs {
      inherit system; config.allowUnfree = true; config.cudaSupport = true;
    };
  in {
    devShell.${system} = pkgs.mkShell {
      buildInputs = with pkgs; [
        nixpkgs-python.packages.${system}."3.11.8"
        pkgs.nodejs_20
        uv
        cudaPackages.cudatoolkit
        cudaPackages.cudnn
      ];

      shellHook = ''
        export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
        export LD_LIBRARY_PATH=${
          pkgs.lib.makeLibraryPath [
            pkgs.zlib
            pkgs.stdenv.cc.cc
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ]
        }:/run/opengl-driver/lib:$LD_LIBRARY_PATH

        export NPM_CONFIG_PREFIX="$HOME/.npm-global"
        export PATH="$HOME/.npm-global/bin:$PATH"
        export UV_SYSTEM_PYTHON=1

        if [ ! -f "$HOME/.npm-global/bin/claude" ]; then
          npm install -g @anthropic-ai/claude-code
        fi
      '';
    };
  };
}
