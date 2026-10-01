# %% [markdown]
# # Shortcut check for the SYNTHETIC generators (T02)
#
# The 2-D figures can overlap while the 29-D data does not. This script asks a strong,
# nonlinear model (a random forest) to separate each attack class from its benign
# look-alike, at difficulty 1 with no label noise (so every error comes from real overlap in
# the packets, not from flipped labels).
#
# Target: accuracy below 0.95 for every pair. A pair above it is either real traffic that a
# single flow does show, or a generator artifact (a rule written for one class only) that
# must be fixed. The top features point at the cause.
#
# A whole-pair score can hide an artifact that only part of a class has, so the script also
# checks the hard subset of each pair: the flows that look most alike per flow.
#
# Run: uv run python explore/shortcut_check.py
# The numbers describe SYNTHETIC data and the generator rules, not detection performance.

# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from neural_ids.schema import FEATURE_NAMES
from neural_ids.synthetic import ATTACK_CLASSES, NORMAL_CLASSES, make_dataset

PAIRS = (("ssh_bruteforce", "ssh_session"), ("syn_scan", "web"), ("udp_flood", "dns"))
N_FLOWS, SEED, FOLDS, TARGET = 12_000, 42, 5, 0.95  # 2000 flows per class

# %% One dataset, true classes only (label_noise=0), every class 2000 flows.
df = make_dataset(N_FLOWS, attack_fraction=0.5, difficulty=1.0, seed=SEED, label_noise=0.0)
assert set(df["label"]) == set(NORMAL_CLASSES + ATTACK_CLASSES)


# %%
# Hard subsets: the per-flow look-alikes inside each pair (masks on features only).
SUBSETS = {
    "ssh_bruteforce": ("<= 20 packets", lambda d: d["fwd_packets"] + d["bwd_packets"] <= 20),
    "syn_scan": ("<= 3 packets", lambda d: d["fwd_packets"] + d["bwd_packets"] <= 3),
    "udp_flood": ("no reply", lambda d: d["bwd_packets"] == 0),
}


def check_pair(attack: str, benign: str, subset: bool = False) -> dict[str, object]:
    """5-fold CV accuracy of a random forest on one pair, and its top 3 features.

    With `subset`, only the pair's hard subset (`SUBSETS`). Feature importances come from one
    fit on all rows used (they explain the model, they are not a score).
    Cost: O(folds * trees * n log n) for n rows.
    """
    pair = df[df["label"].isin([attack, benign])]
    name = f"{attack} vs {benign}"
    if subset:
        description, mask = SUBSETS[attack]
        pair = pair[mask(pair)]
        name += f" [{description}]"
    x = pair[list(FEATURE_NAMES)].to_numpy()  # x: (n, 29)
    y = (pair["label"] == attack).to_numpy().astype(int)  # y: (n,)
    forest = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    accuracy = cross_val_score(forest, x, y, cv=folds, scoring="accuracy")  # (FOLDS,)
    importance = pd.Series(forest.fit(x, y).feature_importances_, index=FEATURE_NAMES)
    top = importance.sort_values(ascending=False).head(3)
    return {
        "pair": name,
        "rows": f"{int(y.sum())}/{int(len(y) - y.sum())}",
        "accuracy": accuracy.mean(),
        "std": accuracy.std(),
        "below_target": accuracy.mean() < TARGET,
        "top_3_features": ", ".join(f"{feature} ({value:.2f})" for feature, value in top.items()),
    }


table = pd.DataFrame([check_pair(a, b) for a, b in PAIRS])
hard = pd.DataFrame([check_pair(a, b, subset=True) for a, b in PAIRS])
print(f"SYNTHETIC data, difficulty 1, label_noise 0, {FOLDS}-fold CV, random forest")
print(f"target: accuracy < {TARGET} for every pair")
with pd.option_context("display.width", 200, "display.max_colwidth", 120):
    print(table.round(3).to_string(index=False))
    print("\nHard subsets (rows = attack/benign; informational, no target):")
    print(hard.drop(columns="below_target").round(3).to_string(index=False))
