# main's history was rewritten on 2026-10-30

At the owner's request, the 13 commit messages on `main` that ended with a `Claude-Session: https://claude.ai/code/...`
line were rewritten without that line. Claude's use in this project is disclosed in the README.

**What changed and what didn't:**
- Only those 13 messages changed. Every commit's tree (its files), author, committer and dates are identical to
  before.
- Every descendant of a rewritten commit got a new ID: 354 of the 360 commits.
- GitHub's signatures on its own 72 web-merge commits could not be carried over, so those merges no longer show
  "Verified".
- The tip moved from `f0f88e9` to `5902f72`; the push used a lease on the old tip.

**Old commit IDs.** Protocols, manifests, evidence bundles and verifiers written before this date cite the *old* IDs.
[`commit_id_map_2026-10-30.tsv`](commit_id_map_2026-10-30.tsv) maps each old ID to its new one (old, then new). The
old commits remain reachable through the older remote branches, the three `ti-tnn-*` release tags and the
pull-request refs, so `git show <old id>` still works in a full clone.
