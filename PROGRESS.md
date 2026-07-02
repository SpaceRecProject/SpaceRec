TODO
- [x] Confirm current workspace and GitHub CLI authentication.
- [x] Confirm repository is not already initialized as git.
- [x] Add git ignore rules for local data, results, caches, and large archives.
- [x] Update README with GitHub data availability notes for Google Drive.
- [ ] Initialize local git repository and commit code-only package.
- [ ] Create private GitHub repository `SpaceRec` and push.
- [ ] Verify GitHub remote and that `data/` and `results/` are not tracked.

Progress Log

2026-07-02
- Working directory confirmed: `/net/dali/home/chikina/shared_data/SpaceRec`.
- Host confirmed: `g017`; `SLURM_JOB_ID` and `CUDA_VISIBLE_DEVICES` are unset. This task is limited to git and documentation operations; no CUDA or model computation is being run.
- `gh` is installed and authenticated to GitHub as `OliverWang0908`.
- Current directory is not a git repository.
- Existing root files before setup: `README.md`; no `.gitignore` or `PROGRESS.md`.
- Size check: `data/` is 2.2G, `results/` is 24G, `spacerec/` is 2.5M. `data/` and `results/` must stay out of git.
- Cache directories found under `spacerec/**/__pycache__`; they will be ignored.
- Added `.gitignore` excluding `data/`, `results/`, large archives, Python caches, notebook checkpoints, and local environment files.
- Updated `README.md` with a data/results availability section using a Google Drive placeholder link. A real shared folder URL still needs to be added after upload.
- Initialized local git repository and renamed the initial branch to `main`.
- `git add -n .` showed only code, notebook, README, PROGRESS, `.gitignore`, and `.vscode/settings.json`; `data/`, `results/`, and cache directories were not included.
- Initial commit created: `1bd7741 Initial SpaceRec slim package`.
- Attempt to create `OliverWang0908/SpaceRec` failed because the repository already existed.
- Existing GitHub repository checked: `OliverWang0908/SpaceRec` is private, non-empty, and current user has ADMIN permission.
- User instructed to delete the existing `OliverWang0908/SpaceRec` repository and recreate it from the current local package.
- Attempted `gh repo delete OliverWang0908/SpaceRec --yes`; this `gh` version uses `--confirm` instead.
- Attempted `gh repo delete OliverWang0908/SpaceRec --confirm`; deletion failed because the current token lacks `delete_repo` scope.
- User manually deleted `OliverWang0908/SpaceRec` on GitHub.
- Confirmed `gh repo view OliverWang0908/SpaceRec` no longer resolves. Cleaned up the interrupted `gh auth refresh` process.
