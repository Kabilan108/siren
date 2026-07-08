---
name: cut-release
description: Cut a Siren release by choosing the appropriate semver bump, updating the project version with uv, adding a CHANGELOG.md entry, and starting the local GPU Docker image build/push in tmux with logs captured under /tmp. Use when the user asks to cut, prepare, publish, or release a new Siren Docker image.
---

# Cut Release

## Overview

Prepare a Siren release from the current repo state, then start the Docker Hub GPU image build asynchronously in tmux. Keep the release metadata changes separate from the long-running build process.

## Workflow

1. Inspect the pending changes and recent commits:
   - Run `git status --short`.
   - Review the relevant diff with `git diff` and recent history with `git log --oneline --decorate -n 20`.
   - Decide the bump level: `major` for breaking changes, `minor` for new compatible behavior, `patch` for fixes, dependency refreshes, and small internal changes.
   - If the bump level is genuinely ambiguous, ask the user before changing files.

2. Bump the version with uv:
   - Run `uv version --bump=<major|minor|patch>`.
   - Capture the new version with `uv version --short`.
   - Use the bare semver value from uv for Docker tags, for example `0.2.0`, not `v0.2.0`.

3. Update `CHANGELOG.md`:
   - Create it if missing.
   - Add a new entry at the top for the new version using the current date.
   - Summarize user-visible changes and notable maintenance changes from the diff/history.
   - Keep entries concise and factual.

4. Validate before starting the image build:
   - Run the fastest relevant checks available for the release changes.
   - At minimum, run `uv version --short` and verify the build helper with `bash -n scripts/build-push-gpu-image`.
   - Do not start the Docker push if the release metadata is inconsistent.

5. Start the GPU image build/push in tmux:
   - Require an explicit Docker Hub repository from the user or existing repo context before publishing.
   - Write logs to a temp path such as `/tmp/siren-release-<version>-<timestamp>.log`.
   - Run the build through `bash -lc` with `set -o pipefail` so a failed build is not hidden by `tee`.
   - If running inside tmux (`$TMUX` is non-empty), create a new pane in the current session.
   - If not running inside tmux, create a throwaway detached tmux session that runs the build and exits when the command succeeds. Keep the shell open on failure.
   - In both cases, tell the user the tmux target and log path.

## Tmux Commands

Inside tmux:

```bash
log_path="/tmp/siren-release-${version}-$(date +%Y%m%d-%H%M%S).log"
tmux split-window -h "cd '$PWD' && bash -lc 'set -o pipefail; scripts/build-push-gpu-image \"$repository\" latest 2>&1 | tee \"$log_path\"; status=\${PIPESTATUS[0]}; if (( status == 0 )); then exit 0; fi; printf \"Build failed with status %s. Log: %s\n\" \"\$status\" \"$log_path\"; exec bash'"
```

Outside tmux:

```bash
log_path="/tmp/siren-release-${version}-$(date +%Y%m%d-%H%M%S).log"
session="siren-release-${version//./-}"
tmux new-session -d -s "$session" "cd '$PWD' && bash -lc 'set -o pipefail; scripts/build-push-gpu-image \"$repository\" latest 2>&1 | tee \"$log_path\"; status=\${PIPESTATUS[0]}; if (( status == 0 )); then exit 0; fi; printf \"Build failed with status %s. Log: %s\n\" \"\$status\" \"$log_path\"; exec bash'"
```

The throwaway session disappears after a successful build because the shell command exits. On failure, inspect it with `tmux attach -t "$session"` or read the log path.

## Publish Guardrails

- Do not run `docker login` for the user. If Docker Hub auth is missing, report that `docker login` is required.
- Do not publish to a guessed namespace. Use only a repository provided by the user or already documented in the repo.
- Do not create git commits or git tags unless the user explicitly asks.
- Leave the build running in tmux if it is still active when the response is ready, and include the follow-up command to inspect it.
