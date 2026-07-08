{
  description = "OpenAI compatible STT server with Faster Whisper and Parakeet backends";

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
        '';
      };

      nixosModules.default =
        {
          config,
          lib,
          pkgs,
          ...
        }:
        let
          cfg = config.services.siren;
          # .python-version pins 3.13.2 exactly and uv enforces it
          python = nixpkgs-python.packages.${pkgs.stdenv.hostPlatform.system}."3.13.2";
          # same library set the devshell uses; wheels (torch, nemo) load these
          # at runtime, with the host driver coming from /run/opengl-driver
          runtimeLibs = lib.makeLibraryPath [
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.libsndfile
          ];
        in
        {
          options.services.siren = {
            enable = lib.mkEnableOption "siren speech-to-text server";

            host = lib.mkOption {
              type = lib.types.str;
              default = "127.0.0.1";
            };

            port = lib.mkOption {
              type = lib.types.port;
              default = 8301;
            };

            environmentFile = lib.mkOption {
              type = lib.types.nullOr lib.types.path;
              default = null;
              description = "Environment file providing SIREN_API_KEY.";
            };

            hfHome = lib.mkOption {
              type = lib.types.path;
              default = "/var/lib/siren/huggingface";
              description = "HF_HOME for model downloads and cache.";
            };
          };

          config = lib.mkIf cfg.enable {
            users.users.siren = {
              isSystemUser = true;
              group = "siren";
              home = "/var/lib/siren";
            };
            users.groups.siren = { };

            systemd.services.siren = {
              description = "Siren speech-to-text server";
              wantedBy = [ "multi-user.target" ];
              after = [ "network-online.target" ];
              wants = [ "network-online.target" ];

              path = [
                pkgs.uv
                pkgs.stdenv.cc
                pkgs.ffmpeg
              ];

              environment = {
                HOME = "/var/lib/siren";
                HF_HOME = cfg.hfHome;
                UV_PROJECT_ENVIRONMENT = "/var/lib/siren/venv";
                UV_CACHE_DIR = "/var/lib/siren/uv-cache";
                UV_PYTHON_DOWNLOADS = "never";
                UV_PYTHON = "${python}/bin/python3.13";
                CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
                LD_LIBRARY_PATH = "${runtimeLibs}:/run/opengl-driver/lib";
              };

              serviceConfig = {
                User = "siren";
                Group = "siren";
                StateDirectory = "siren";
                WorkingDirectory = "/var/lib/siren";
                EnvironmentFile = lib.mkIf (cfg.environmentFile != null) cfg.environmentFile;

                # venv rebuild is a no-op when uv.lock is unchanged; first run
                # resolves from the lockfile and may compile the few sdists
                ExecStartPre = "${pkgs.uv}/bin/uv sync --frozen --no-dev --project ${self}";
                ExecStart = "/var/lib/siren/venv/bin/fastapi run ${self}/siren --host ${cfg.host} --port ${toString cfg.port}";
                # cold-cache uv sync in ExecStartPre downloads ~7GB of wheels
                TimeoutStartSec = 1800;

                Restart = "on-failure";
                RestartSec = 5;

                NoNewPrivileges = true;
                PrivateTmp = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                ReadWritePaths = [ cfg.hfHome ];
                SupplementaryGroups = [
                  "video"
                  "render"
                ];
              };
            };
          };
        };
    };
}
