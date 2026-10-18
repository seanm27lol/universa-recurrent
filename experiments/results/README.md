# Development artifacts

`development_microbenchmark.json` is an exploratory local run, not a sealed or
preregistered experiment. It preserves all 27 trial rows, including separate
setup, routing, solve/record, serialization, and checking costs.

The inputs are deliberately tiny. Do not use these numbers to claim general
performance. The benchmark runner refuses to overwrite an existing output.

## TI-TNN

[`ti_tnn/`](ti_tnn/) holds weight-free, byte-identical result exports for the completed TI-TNN
experiments, with a manifest of hashes and declared source versions, and `make_tables.py`, which
rebuilds the reports' tables from them. It is a reviewed exception to keeping run outputs in the
git-ignored `runs/`; see its README.
