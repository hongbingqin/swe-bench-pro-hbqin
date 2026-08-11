  I'm making `groups` optional — defaults to `None`, which means auto-discover. You can still pre-list them if
  you want; backward-compat preserved. When a new group value shows up in `group_labels` during `update()` that
  I haven't seen before, I spin up a fresh deep-copied base metric on the fly. For
  `state_dict`/`load_state_dict` round-trip: if you save after discovering groups and load into a fresh
  empty-groups instance, it restores the discovered groups plus their state. `reset()` and `compute()` operate
  over the full dynamically-discovered set. SubgroupDifference's existing ≥2-groups rule still applies —
  `compute()` raises if fewer than two groups have been seen. Invalid or unrecognized group handling stays
  strict.
