"""Article-level splitting. Articles are assigned to splits BEFORE any text is generated."""
import numpy as np

SPLIT_ORDER = ["S1", "S2", "S3", "S4", "S5"]


def assign_splits(article_ids, sizes, seed):
    """sizes: dict like {"S1": 600, "S2": 200, "S3": 300, "S4": 1000, "S5": 300}.

    Returns {split: [article_id, ...]}. Deterministic given seed; raises if too few articles."""
    ids = sorted(article_ids)
    need = sum(sizes[s] for s in SPLIT_ORDER)
    if len(ids) < need:
        raise ValueError(f"need {need} eligible articles, only {len(ids)} available")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    out, pos = {}, 0
    for s in SPLIT_ORDER:
        k = sizes[s]
        out[s] = [ids[i] for i in order[pos:pos + k]]
        pos += k
    return out
