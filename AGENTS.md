# The rule that must survive every future edit

**Keep this project penetrable for anyone curious to understand it, while grounding
its ideas in known examples from the world and established fields.**

This is an enduring project requirement, not a README decoration.

## Explain in this order

1. A concrete example: what problem is someone trying to solve?
2. Plain-language intuition: what changes, and why might that help?
3. Mathematics: define every symbol and state the assumptions.
4. Code: point to a runnable, small implementation.
5. Evidence: say exactly what the test or certificate establishes.
6. Limits: say what it does not establish; cite primary prior work.

Keep the opening short. Put depth in linked pages. Prefer small tables, diagrams,
worked examples, meaningful variable names, and useful docstrings over slogans.
A reader should not need category theory to run the first example. More advanced
readers should be able to locate precise definitions and proofs.

## Scientific honesty

- Distinguish implemented, tested locally, externally replicated, and proposed.
- The repository contains a classical baseline and an exploratory neural model. Do not treat either as a sealed scientific result.
- A passing test is not a preregistered scientific result.
- Do not call routine projected gradient descent or KKT conditions a new discovery.
- Use direct, shallow fixed-depth, structural, and generator-aware references where applicable. Include setup, routing, logging, checking, and preprocessing in claimed total costs.
- Distinguish logical steps, active update examples, sequential rounds, and wall-clock time.
- Known structure is an assumption. A valid certificate for the wrong structure
  does not establish a useful answer.
- Preserve negative results, ambiguous routing, and budget exhaustion explicitly.
- No high-stakes or general intelligence claims from the toy example.

## Keep distinct guarantees distinct

A trace schema, a property certificate, execution provenance, causal faithfulness,
and real-world task accuracy answer different questions. Do not silently transfer
credit from one to another. Compact mode does not check discarded intermediate
history. Floating-point tolerances are not exact Lean proofs. Hashes identify
artifacts but do not authenticate a remote execution.

## Engineering boundaries

- Keep old repositories and sealed evidence untouched.
- Do not vendor entire research repositories. Pin optional dependencies.
- No private application code, model weights, credentials, or data in public files.
- Treat serialized model checkpoints as trusted-input only; restricted loading is not authentication.
- Independent checkers must not invoke the producer's solve or decomposition.
- Match changes to tests, the relevant explanation, the development audit, and the claim ledger.
- Attach a known-field example and a primary source to each new major concept.
- Never publish or change repository visibility without explicit authorization.
- Never claim a remote exists until a successful remote read confirms it.
