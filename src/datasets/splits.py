# src/datasets/splits.py
"""
Subject-disjoint splits for CASIA-B.

Two protocols are supported:

1) "custom" (used in the thesis and the initial Electronics submission):
   99 train / 15 val / 10 test, from a seeded RandomState(42) shuffle.
   Rationale: provide an independent held-out test set unused during
   model selection.

2) "standard" (added for the Electronics revision, Reviewer 1 R1-1):
   74 train / 50 test, matching the field-standard CASIA-B split used
   by GaitSet, GaitGraph, GaitBase, GaitPT, and MAST-GCN.
   Subjects 1-74 are training; 75-124 are test.
   No separate validation set under this protocol; checkpoint selection
   uses the final training step, as in those baselines.
"""

# ---- Custom protocol (thesis, 99/15/10) ----
TEST_SUBJECTS = frozenset([25, 41, 56, 73, 76, 78, 80, 81, 100, 110])
VAL_SUBJECTS  = frozenset([2, 5, 8, 12, 17, 19, 31, 37, 43, 63, 67, 83, 91, 108, 114])
TRAIN_SUBJECTS = frozenset(range(1, 125)) - TEST_SUBJECTS - VAL_SUBJECTS

# ---- Standard CASIA-B protocol (74/50) ----
STANDARD_TRAIN_SUBJECTS = frozenset(range(1, 75))     # subjects 1-74
STANDARD_TEST_SUBJECTS  = frozenset(range(75, 125))   # subjects 75-124

# ---- Assertions ----
assert len(TEST_SUBJECTS) == 10
assert len(VAL_SUBJECTS) == 15
assert len(TRAIN_SUBJECTS) == 99
assert TEST_SUBJECTS.isdisjoint(VAL_SUBJECTS)
assert TEST_SUBJECTS.isdisjoint(TRAIN_SUBJECTS)
assert VAL_SUBJECTS.isdisjoint(TRAIN_SUBJECTS)
assert TEST_SUBJECTS | VAL_SUBJECTS | TRAIN_SUBJECTS == frozenset(range(1, 125))
assert len(STANDARD_TRAIN_SUBJECTS) == 74
assert len(STANDARD_TEST_SUBJECTS) == 50
assert STANDARD_TRAIN_SUBJECTS.isdisjoint(STANDARD_TEST_SUBJECTS)
assert STANDARD_TRAIN_SUBJECTS | STANDARD_TEST_SUBJECTS == frozenset(range(1, 125))


def get_split(protocol: str = "custom"):
    """Return (train, val, test) subject frozensets for the given protocol.

    Standard protocol has empty val; callers should skip val-based checkpoint
    selection and use the final-step checkpoint for test evaluation.
    """
    p = protocol.strip().lower()
    if p == "custom":
        return TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS
    if p == "standard":
        return STANDARD_TRAIN_SUBJECTS, frozenset(), STANDARD_TEST_SUBJECTS
    raise ValueError(f"Unknown protocol: {protocol!r}. Use 'custom' or 'standard'.")