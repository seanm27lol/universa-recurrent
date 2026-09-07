# Start here: five sensors, two independent choices

Suppose flow circulates around a square, with one diagonal pipe. A negative flow
means motion opposite to the arrow; this toy model does not enforce pipe capacity.

```text
          e0
      0 ─────→ 1
      ↑ ╲      │
   e3 │  ╲ e4  │ e1
      │   ↘    ↓
      3 ←───── 2
          e2
```

The demo reads three noisy sensor values and holds two other readings for choosing
between candidate conservation rules. The hidden synthetic truth is used only
for the final illustrative error report, never for routing or solving.

## 1. What does “structure” mean?

At each junction, incoming flow equals outgoing flow. Those relationships rule out
many combinations of the five edge values. Only two independent numbers are
needed to describe every balanced circulation on this graph.

**Important:** this is conservation alone, not a simulator of real pipes or a full
circuit. Pumps, resistances, pressure, capacity, and accumulation are not modeled.
See the [real-world analogy and its boundary](real_world_connections.md).

## 2. What does “latent” mean here?

Instead of carrying five edge values as the working coordinates, the solver
carries two cycle coordinates. They determine all five values. These coordinates
are calculated from the known constraint, not learned by a neural network.

Changing coordinate systems is familiar: a map can describe a location by street
address or by coordinates. Here the useful coordinates eliminate invalid flows.
The analogy concerns representation, not a claim that every representation is
lossless or interchangeable.

## 3. What does “recurrence” mean?

Start with an estimate, nudge it toward the measurements, and keep it balanced.
Reuse the same update rule and the current state. Stop when the remaining
optimization residual is small, or when the iteration budget is used up.

The implementation is projected gradient descent, a known optimization method.
We first need a correct baseline before replacing parts with a learned updater.
Running one exact projection repeatedly would not help: P(P(v))=P(v).

## 4. What does “witness” mean?

A witness is extra information that makes a claim easier to check. In a maze,
a route lets someone check that you reached the exit without searching for a route
themselves. Here a small vector helps check that a projection or final constrained
estimate satisfies particular equations.

It does not prove the sensors were honest, the structure was appropriate, or the
runtime actually followed that route on another machine.

## 5. What does “Lingua” add?

It labels the real operations and their evidence in a readable format. A JSON trace
can say “measurement-fitting step under this constraint” and carry a numerical
witness. It must not invent “the model realized...” explanations.

Next: [run the README example](../README.md#quickstart), inspect
[the retained output](example_run.txt), then read [the equations](mathematics.md).
