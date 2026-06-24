# codimango/skorch-wlm-transformer

## Description

Adds a **Transformer** language model as a selectable option in skorch's
`examples/word_language_model`, alongside the existing RNN. The example's custom
`Net` (a `skorch.NeuralNet` subclass) is built around a stateful RNN
(`init_hidden` + `forward(input, hidden)`), so the core engineering challenge is
making the training loop work with a **stateless** model that has no hidden state.

- **Repo:** `skorch-dev/skorch` @ `e769a2b89dc6e946dfaf42654184aa01b7025964`
- **Language:** Python
- **Files touched by the reference solution:** `examples/word_language_model/{model.py, net.py, train.py}`

## Reference solution (oracle)

- `model.py`: new `TransformerModel` (+ `PositionalEncoding`) — embedding +
  positional encoding + `nn.TransformerEncoder` + linear decoder; `forward(input)`
  with a causal mask, returns `(seq, batch, ntoken)`; no `init_hidden`.
- `net.py`: `Net` branches on whether the module is stateful (`init_hidden`
  present) so a stateless model is called as `module_(X)`; RNN path preserved.
- `train.py`: `--model {rnn, transformer}` selector.

## Completion Rates

_To be populated after Avocado calibration (`codimango bench run -a metacode -m
meta/avocado_dvsc_tester -k 5`)._

| Agent | Model | Attempts | Pass rate |
|-------|-------|----------|-----------|
| oracle | oracle | 1 | 1.000 |

## Model Analysis

_To be populated after calibration._ Predicted failure mode: agents producing a
*reasonable but non-conforming* Transformer (different class name or a
`forward(input, hidden)` signature) and failing `fail_to_pass`. The `net.py`
stateless-branch integration is the non-copyable part (a verbatim
`pytorch/examples` Transformer passes the forward test but fails the
train-through-`Net` test).

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

## Notes

`pass_to_pass` (`test_rnn_still_trains`) guards the existing RNN/LSTM path
(`RNNModel(rnn_type='LSTM')`), which already supports LSTM — i.e. LSTM is a
configuration of the existing model, not a new class.
