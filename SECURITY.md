# Security

## Model checkpoints

Treat `.pt`, `.pth`, and other serialized model files as **trusted input only**.
This project requires PyTorch 2.10 or newer and uses `weights_only=True`, but
neither choice authenticates an unknown file. Prefer checkpoints you trained
locally, verify expected SHA-256 values, and use an isolated environment when
provenance is uncertain.

Lingua checkpoint binding proves only that a record names the same local bytes.
It is not a signature and does not attest that a remote machine executed them.

## Reporting

Please use GitHub's private security-advisory flow for a suspected vulnerability.
Do not include credentials, private data, or weaponized public demonstrations in
an issue.
