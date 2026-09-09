# Contributing

Start with [AGENTS.md](AGENTS.md). It states the project's permanent accessibility,
evidence, and privacy rules for humans and coding assistants.

A good change answers four questions: **What problem does this solve? Where does
this idea already appear? What can the reader run? What has actually been checked?**

## Development

```bash
python -m pip install -e ".[test,neural]"
python -m pytest -q
python -m compileall -q src tests examples experiments
```

Keep the small CPU example working. Add a failure test as well as a success test.
For a new operation, document its input/output types, assumptions, numerical
semantics, witness format, checker cost, and limits. A public technical feature
should be accompanied by an ordinary-language example.

## Evidence and experiments

Do not overwrite previous artifacts. Every timed result must record the machine,
software versions, input sizes, seeds, repetitions, warmup, what was timed, and
what was excluded. A witness checker's runtime is not the solver's runtime. A logical halt is not automatically skipped device work. Report the strongest fixed depth, not only the maximum-depth control. Experiments discovered during development are exploratory, not preregistered.

## Contributions involving earlier projects

Keep adapters small and dependencies explicit. Reuse licensed code only with its
required notices. Never import private application material into this public
research scaffold. A textual mention of a related result is not an executed
integration, and it must not be described as one.
