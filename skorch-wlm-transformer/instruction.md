Add a Transformer-based model as an additional training option, using the same data pipeline and training flow as the existing RNN model. The implementation should integrate cleanly with the current codebase (including model selection and training interfaces) and clearly document any architectural assumptions and required hyperparameters.

For positional information in the Transformer, do not use traditional sinusoidal (cos/sin) positional embeddings. Instead, use ALiBi (Attention with Linear Biases) by adding a distance-based linear bias to the attention scores. This will allow the model to learn positional information in a more flexible and efficient manner, without the need for explicit sinusoidal embeddings.

ALiBi replaces position embeddings with a simple distance-based attention bias. You **don’t add any positional embeddings anywhere**; instead, in each attention head you **add a fixed, non-learned linear penalty after the query·key dot product**: for the *i*‑th query attending over keys 1…i, you add
**m · [−(i−1), …, −1, 0]**, so farther keys get more negative bias (lower attention).

Each head uses its own slope **m**, and the slopes are set **once** as a **geometric sequence** (for *n* heads starting at **2^(−8/n)** with the same ratio), so different heads penalize distance at different rates. This builds in a **recency inductive bias** and generally works across domains/sizes without re-tuning.
