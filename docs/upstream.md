# Upstream boundaries and provenance

The base demo is standalone NumPy code, with original implementations of standard
linear algebra and optimization. It does not vendor or silently modify existing
research repos.

## Optional Universa adapter

- Repository: `seanm27lol/Universa`
- Pinned commit: `132fdf8d5ebe87deb3d6cb597c8ab77407176918`
- Inspected public API: `universa.operators.nullspace_basis`
- Install: `python -m pip install -e ".[universa]"`
- Example: `from universa_recurrent.structures.universa_adapter import compile_from_universa`
- The adapter checks the upstream and local subspaces agree numerically.
- License: upstream Apache-2.0; it remains a separate dependency.

The base tests skip the adapter when Universa is absent. An adapter test passing
against a fake module is not sufficient; record when the actual pinned dependency
was exercised. [Local validation](local_validation.md) states what happened here.

## Research references, not runtime dependencies

HOMYMOLY and Applied CMCM are linked for their evidence and questions. Their data,
models, and entire codebases are not dependencies. The quantum/ZX rewriting and
Lean layers are not claimed to implement neural verification in this repo.

## Publishing boundary

This repo is intended to be public. No private application source, weights, data,
logs, or secrets should ever enter it. The publisher targets only
`seanm27lol/universa-recurrent`, refuses an unexpected authenticated account, and
refuses an existing remote rather than overwrite it. It creates no changes in
upstream repos. The locally prepared archive is not evidence that a GitHub remote
has already been created.
