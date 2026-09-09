# GitHub is the source of releases

Use reviewed Git commits and pull them onto the DGX. Do not run the retired
v0.4 overlay installers or edit a checksum merely to make an installer proceed.

For a clean checkout:

```bash
git pull --ff-only
python -m pip install -e ".[test,neural]"
python scripts/check_release.py
python -m pytest -q
```

For the one-time recovery from a failed installer, first preserve the unfinished
files with `git stash push --include-untracked -m "before GitHub release recovery"`.
Then pull. Keep that stash; do not blindly pop obsolete installer files over the
repaired release. Ignored checkpoints, runs, and virtual environments are not
included by `--include-untracked` (unlike `--all`). Never use `git clean -fdx`.
If a fast-forward fails, stop and reconcile rather than force or reset.

A release should pass source tests, dedicated neural tests, version/module/CLI
checks, and an independently installed wheel smoke check. Only then update main.
The previous scientific protocols and result artifacts are not release manifests
and must not be rewritten to describe a different run.
