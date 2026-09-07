# Publishing this prepared repository

The archive is a local project, not an already-created GitHub remote. The connected
GitHub tools used during preparation exposed read actions, not a write/create
operation; no usable GitHub CLI authentication was present in that environment.

After extracting the archive, run this **inside `universa-recurrent/`**:

```bash
bash scripts/publish.sh
```

It uses GitHub CLI's documented
[`gh repo create --source . --public --push`](https://cli.github.com/manual/gh_repo_create).
It targets only `seanm27lol/universa-recurrent`, verifies the authenticated user,
and refuses an existing repository, existing remote, or unreviewed local changes.
It never overwrites another repo or changes another repo's visibility.

If GitHub CLI is missing on a Mac with Homebrew:

```bash
brew install gh
gh auth login --hostname github.com
bash scripts/publish.sh
```

Do not paste access tokens into a chat or commit them. Authenticate through the
CLI's local/browser flow. If repository creation succeeds but pushing fails, the
remote may exist without files: inspect it and resolve the reported permission or
network issue rather than rerun blindly. Normal publication requires repository
write access; workflow-file pushes may also require the corresponding permission.

The script creates a local initial commit when Git metadata is absent. Its fallback
author is `Project bootstrap <bootstrap@localhost>` so it does not invent your
personal commit identity. Configure your normal Git identity for subsequent work.
