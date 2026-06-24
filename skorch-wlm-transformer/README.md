# codimango/skorch-wlm-transformer

## Description

Adds a **Transformer** language model as a selectable option in skorch's
`examples/word_language_model`, alongside the existing RNN. Two things make this
non-trivial:
1. The example's custom `Net` (a `skorch.NeuralNet` subclass) is built around a
   stateful RNN (`init_hidden` + `forward(input, hidden)`), so the training loop
   has to be made to work with a **stateless** model that has no hidden state.
2. Positional information must use **ALiBi** (Attention with Linear Biases) — a
   per-head linear distance bias added to the attention scores — **not** the
   standard sinusoidal positional encoding. This is what keeps the task off the
   verbatim `pytorch/examples` / PyTorch-tutorial recall path.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python
- **Files touched by the reference solution:** `examples/word_language_model/{model.py, net.py, train.py}`

## Reference solution (oracle)

- `model.py`: new `TransformerModel` — embedding + `nn.TransformerEncoder` +
  linear decoder; **ALiBi** positional bias (per-head linear distance penalty
  added to attention scores via the float attention mask; no positional
  embeddings); `forward(input)` with a causal mask, returns `(seq, batch,
  ntoken)`; no `init_hidden`.
- `net.py`: `Net` branches on whether the module is stateful (`init_hidden`
  present) so a stateless model is called as `module_(X)`; RNN path preserved.
- `train.py`: `--model {rnn, transformer}` selector.

## Completion Rates

Measured by the platform during validation (commit `c2c9f65`):

| Agent | Model | Attempts | Pass | Pass rate |
|-------|-------|----------|------|-----------|
| oracle | oracle | 3 | 3 | 1.000 |
| metacode | avocado_dvsc_tester | 5 | 0 | 0.000 |
| claude-code | claude-opus-4-6 | 5 | 1 | 0.200 |
| (aux) | gpt-5.5 | 5 | 1 | 0.200 |

Balance gate: **passed** — avocado not trivial (0/5) and ≥1 agent solved
(opus 1/5, gpt 1/5).

## Model Analysis

The difficulty comes from two compounding requirements, neither copyable verbatim:
**(1)** integrating a *stateless* Transformer into a training loop (`Net`) that is
written around RNN hidden state, and **(2)** implementing **ALiBi** instead of
sinusoidal positional encoding.

- **avocado 0/5, opus 1/5, gpt 1/5** — a genuine difficulty gradient: the stronger
  paths solve it occasionally; avocado never does. Not too easy (avocado 0/5), not
  unsolvable (opus and gpt each solve it).
- **Why it isn't a recall task:** a verbatim `pytorch/examples` Transformer (with
  sinusoidal `PositionalEncoding`) fails — it would fail the ALiBi enforcement test
  (`test_no_absolute_positional_encoding`) and the stateless-`Net` integration. The
  agent must implement ALiBi's per-head linear bias and wire a no-hidden-state model
  through the existing loop.
- **Note on gpt-5.5 trials:** some gpt trials across runs failed on agent-harness
  infrastructure (`codex` non-zero exit / rate-limit), not reasoning; on clean runs
  gpt solves it (1–2/5). Discount infra exits when reading gpt's rate.

## Anti-Cheating Analysis

- **Tests are verifier-only.** The agent sees only `instruction.md` at solve time;
  the test file is applied via `test_patch` at verify time, so the required
  interface cannot be read off the tests.
- **fail_to_pass executes the patched code** — imports the patched `model.py` and
  calls the patched `Net.train_step`; not a static-artifact check.
- **nop fails / oracle passes.** At base commit the transformer tests fail
  (`ImportError`) and the RNN regression test passes; after the solution all pass.
- **No oracle/ground-truth in agent-readable paths** — the Dockerfile only clones
  the repo and pip-installs pinned deps; no `COPY`/`ADD` of reference data.
- **Behavioral assertions** — shapes and finite-loss, not implementation-internal
  attribute names (any valid implementation passes).
- **ALiBi is enforced behaviorally, leniently.** `test_no_absolute_positional_encoding`
  feeds a constant-token sequence and asserts position-invariant logits — true for
  ALiBi (and any relative scheme), false for sinusoidal/learned absolute encodings.
  It accepts any relative-bias implementation while blocking the sinusoidal
  copy-paste, so the spec's ALiBi requirement isn't "too lenient."

## Notes

`pass_to_pass` (`test_rnn_still_trains`) guards the existing RNN/LSTM path
(`RNNModel(rnn_type='LSTM')`), which already supports LSTM — i.e. LSTM is a
configuration of the existing model, not a new class.
