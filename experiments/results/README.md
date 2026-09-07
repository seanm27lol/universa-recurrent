# Development artifacts

`development_microbenchmark.json` is an exploratory local run, not a sealed or
preregistered experiment. It preserves all 27 trial rows, including separate
setup, routing, solve/record, serialization, and checking costs.

The inputs are deliberately tiny. Do not use these numbers to claim general
performance. The benchmark runner refuses to overwrite an existing output.
