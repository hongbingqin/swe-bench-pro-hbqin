Precision, Recall, and Fβ all have a divide-by-zero edge case: precision blows up when nothing was predicted for a class, recall blows up when a class has no actual samples. Right now we just hand back 0 in those spots. That's a defensible default, but it's not the only reasonable one — sometimes you want 1 ("I made no mistakes on an empty set, so that's perfect") and sometimes you want NaN ("there's genuinely no answer here, don't silently paper over it").

So: give all three metrics a zero_division kwarg. It accepts 0, 1, or NaN. Default stays 0 so nothing breaks for existing users.

It must work correctly across all average modes, matching sklearn.

The semantics should line up exactly with sklearn's precision_score / recall_score / fbeta_score for the same zero_division value. Tests will verify against sklearn as ground truth.
