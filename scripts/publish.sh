#!/usr/bin/env bash
# Publishes ONLY this prepared project. It never changes any upstream repository.
set -euo pipefail
export GH_HOST=github.com
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TARGET="seanm27lol/universa-recurrent"
for tool in git gh; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing $tool. Install Git and GitHub CLI before publishing." >&2
    echo "On macOS with Homebrew: brew install gh" >&2
    exit 1
  fi
done
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  echo "Authenticate locally first: gh auth login --hostname github.com" >&2
  exit 1
fi
LOGIN="$(gh api --hostname github.com user --jq .login)"
if [[ "$LOGIN" != "seanm27lol" ]]; then
  echo "Refusing: authenticated as $LOGIN, expected seanm27lol." >&2
  exit 1
fi
if gh repo view "$TARGET" >/dev/null 2>&1; then
  echo "Refusing: $TARGET already exists. Nothing was overwritten." >&2
  exit 1
fi
if [[ ! -d .git ]]; then
  git init -b main
  # Explicit allowlist: do not sweep unrelated local files into a public commit.
  git add -- README.md AGENTS.md CONTRIBUTING.md LICENSE NOTICE Makefile     pyproject.toml .gitignore .github src tests examples experiments docs scripts
  git -c user.name="Project bootstrap" -c user.email="bootstrap@localhost"     commit -m "Add readable structured-recurrence baseline, witnesses, and tests"
fi
if [[ "$(git rev-parse --show-toplevel)" != "$ROOT" ]]; then
  echo "Refusing: Git root does not match this project." >&2
  exit 1
fi
if [[ -n "$(git remote)" ]]; then
  echo "Refusing: an existing remote is configured; review it before publishing." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing: review and commit local changes before making the project public." >&2
  exit 1
fi
echo "Publishing $TARGET as PUBLIC. No other repository will be changed."
gh repo create "$TARGET" --public --source . --remote origin --push   --description "Readable structured recurrence with compact, checkable mathematical witnesses."
gh repo view "$TARGET" --json nameWithOwner,url,isPrivate
