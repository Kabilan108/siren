---
name: cut-release
description: Cut a Siren release - choose the semver bump, update the version with uv, add a CHANGELOG.md entry, and hand off deployment (flake input bump on sietch; optional Docker image build/push for non-nix hosts). Use when the user asks to cut, prepare, publish, or release Siren.
---

# Cut Release

Prepare a release from the current repo state. Siren's primary deployment
(sietch) consumes this repo as a **nix flake input** — a release is a version
bump + changelog + pushed commit. The Docker image is an optional secondary
artifact for non-nix hosts.

## Workflow

1. Inspect what's shipping:
   - `git status --short`, `git diff`, `git log --oneline -n 20`
   - Decide the bump: `major` breaking, `minor` new compatible behavior,
     `patch` fixes/deps/internal. Ask the user only if genuinely ambiguous.

2. Bump and document:
   - `uv version --bump=<level>`; capture `uv version --short`.
   - Add a dated entry at the top of `CHANGELOG.md`: concise, factual,
     user-visible changes first.

3. Sanity-check the flake (sietch consumes `nixosModules.default`):
   - `nix flake check --no-build` (or at minimum `nix flake show`).

4. Commit and push when the user confirms (a release only reaches sietch via
   the pushed commit). Do not tag unless asked.

5. Deploy to sietch (from `~/dotfiles`):
   - `nix flake update siren`
   - build check: `nix build .#nixosConfigurations.sietch.config.system.build.toplevel --no-link`
   - user runs: `sudo nixos-rebuild switch --flake ~/dotfiles#sietch`
   - verify: `systemctl is-active siren` and `curl -s -o /dev/null -w '%{http_code}' https://siren.sole-pierce.ts.net/docs`

## Optional: Docker image (non-nix hosts)

Only when the user asks for an image release. Build/push is long — run it in
tmux with logs under /tmp:

```bash
log_path="/tmp/siren-release-${version}-$(date +%Y%m%d-%H%M%S).log"
tmux split-window -h "cd '$PWD' && bash -lc 'set -o pipefail; scripts/build-push-gpu-image kabilan108/siren latest 2>&1 | tee \"$log_path\"; status=\${PIPESTATUS[0]}; if (( status == 0 )); then exit 0; fi; printf \"Build failed with status %s. Log: %s\n\" \"\$status\" \"$log_path\"; exec bash'"
```

(Outside tmux: use a throwaway detached session instead of a split.) Tell the
user the tmux target and log path, and leave failed builds open for inspection.

## Guardrails

- Never run `docker login` for the user; report if auth is missing.
- Publish only to `kabilan108/siren` (or a repo the user names).
- No commits/tags without the user's go-ahead.
