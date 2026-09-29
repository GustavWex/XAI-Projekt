"""
Framework for Assignment 1: Auditing and Explaining Diabetes Risk Prediction Models.

The notebook (projekt.ipynb) tells the story; this module holds the reusable parts:

    1. Data      - loading, codebook (labels, actionability, sensitivity), splitting
    2. Models    - candidate model factory + evaluation (overall and per subgroup)
    3. XAI       - permutation importance, SHAP, LIME behind one common interface
    4. Validation- method agreement, deletion (faithfulness) test, LIME stability,
                   "safety" check of the direction of actionable advice
    5. Actions   - what-if / counterfactual search over actionable features only
    6. Plot style- shared colours and a matplotlib style so all figures look alike

All SHAP / LIME attributions are expressed in *probability* units (percentage
points of predicted risk), so they are comparable across models and easy to
translate for non-technical stakeholders.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "Diabetes_binary"
DATA_PATH = Path(__file__).parent / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
FIG_DIR = Path(__file__).parent / "figures"


# =============================================================================
# 1. DATA + CODEBOOK
# =============================================================================
@dataclass
class Feature:
    label: str              # plain-language name (used in director / patient figures)
    description: str        # codebook meaning (BRFSS 2015)
    kind: str               # "binary" | "ordinal" | "continuous"
    actionable: str         # "yes" | "medical" (changeable with treatment) | "no"
    sensitive: bool = False # protected / socio-economic attribute -> bias audit
    values: dict = field(default_factory=dict)  # code -> meaning, for readable output


AGE_BANDS = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
             8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"}
INCOME_BANDS = {1: "<$10k", 2: "$10-15k", 3: "$15-20k", 4: "$20-25k", 5: "$25-35k",
                6: "$35-50k", 7: "$50-75k", 8: ">$75k"}
EDU_LEVELS = {1: "No school", 2: "Elementary", 3: "Some high school", 4: "High school grad",
              5: "Some college", 6: "College grad"}
GENHLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
YES_NO = {0: "No", 1: "Yes"}

CODEBOOK: dict[str, Feature] = {
    "HighBP":   Feature("High blood pressure", "Ever told by a health professional you have high BP", "binary", "medical", values=YES_NO),
    "HighChol": Feature("High cholesterol", "Ever told your blood cholesterol is high", "binary", "medical", values=YES_NO),
    "CholCheck": Feature("Cholesterol check (5 yrs)", "Cholesterol checked within past 5 years", "binary", "no", values=YES_NO),
    "BMI":      Feature("Body mass index (BMI)", "Body mass index", "continuous", "yes"),
    "Smoker":   Feature("Has smoked (100+ cigarettes)", "Smoked at least 100 cigarettes in entire life", "binary", "no", values=YES_NO),
    "Stroke":   Feature("Has had a stroke", "Ever told you had a stroke", "binary", "no", values=YES_NO),
    "HeartDiseaseorAttack": Feature("Heart disease / heart attack", "Coronary heart disease or myocardial infarction", "binary", "no", values=YES_NO),
    "PhysActivity": Feature("Physically active", "Physical activity in past 30 days (not job)", "binary", "yes", values=YES_NO),
    "Fruits":   Feature("Eats fruit daily", "Consume fruit 1+ times per day", "binary", "yes", values=YES_NO),
    "Veggies":  Feature("Eats vegetables daily", "Consume vegetables 1+ times per day", "binary", "yes", values=YES_NO),
    "HvyAlcoholConsump": Feature("Heavy drinking", "Men >14, women >7 drinks per week", "binary", "yes", values=YES_NO),
    "AnyHealthcare": Feature("Has health insurance", "Any kind of health care coverage", "binary", "no", values=YES_NO),
    "NoDocbcCost": Feature("Skipped doctor due to cost", "Could not see doctor in past 12 months because of cost", "binary", "no", values=YES_NO),
    "GenHlth":  Feature("Self-rated general health", "1=excellent ... 5=poor", "ordinal", "medical", values=GENHLTH),
    "MentHlth": Feature("Days of poor mental health", "Days of poor mental health in past 30 days", "continuous", "medical"),
    "PhysHlth": Feature("Days of poor physical health", "Days of physical illness/injury in past 30 days", "continuous", "medical"),
    "DiffWalk": Feature("Difficulty walking", "Serious difficulty walking or climbing stairs", "binary", "no", values=YES_NO),
    "Sex":      Feature("Sex", "0=female, 1=male", "binary", "no", sensitive=True, values={0: "Female", 1: "Male"}),
    "Age":      Feature("Age group", "13-level age category (5-yr bands)", "ordinal", "no", sensitive=True, values=AGE_BANDS),
    "Education": Feature("Education level", "1=never attended ... 6=college graduate", "ordinal", "no", sensitive=True, values=EDU_LEVELS),
    "Income":   Feature("Income level", "1=<$10k ... 8=>$75k", "ordinal", "no", sensitive=True, values=INCOME_BANDS),
}

FEATURES = list(CODEBOOK)
ACTIONABLE = [f for f, v in CODEBOOK.items() if v.actionable == "yes"]
MEDICAL = [f for f, v in CODEBOOK.items() if v.actionable == "medical"]
SENSITIVE = [f for f, v in CODEBOOK.items() if v.sensitive]
CATEGORICAL = [f for f, v in CODEBOOK.items() if v.kind != "continuous"]

# Direction a *healthy* change goes for each actionable feature. Used to check that
# the model/explanations never reward an unhealthy change (the "safety" check).
HEALTHY_DIRECTION = {"BMI": -1, "PhysActivity": +1, "Fruits": +1, "Veggies": +1,
                     "HvyAlcoholConsump": -1, "HighBP": -1, "HighChol": -1, "GenHlth": -1,
                     "MentHlth": -1, "PhysHlth": -1}


def label(feature: str) -> str:
    return CODEBOOK[feature].label if feature in CODEBOOK else feature


def describe_value(feature: str, value) -> str:
    """Readable value, e.g. describe_value('Age', 9) -> '60-64'."""
    vals = CODEBOOK[feature].values
    return vals.get(int(value), str(value)) if vals else f"{value:g}"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path).astype(int)
    assert list(df.columns) == [TARGET] + FEATURES, "Unexpected columns"
    return df


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1, seed=SEED):
    """Stratified train / validation / test split. Validation is for model selection
    and tuning; the test set is only touched for final numbers and explanations."""
    X, y = df[FEATURES], df[TARGET]
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 2. MODELS + EVALUATION
# =============================================================================
def make_models(seed=SEED) -> dict:
    """Candidate models. They differ in inductive bias:
    - Logistic regression: linear, additive in log-odds, directly interpretable.
    - Random forest: bagged deep trees, non-linear + interactions, low variance.
    - Gradient boosting: sequential shallow trees, usually the strongest on tabular data.
    - Neural network (MLP): smooth non-linear function learned by gradient descent;
      needs scaled inputs, a black box that can only be explained post hoc.
    """
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                                n_jobs=-1, random_state=seed),
        "Gradient boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                                            early_stopping=True, random_state=seed),
        "Neural network": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, learning_rate_init=1e-3,
            batch_size=256, max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=seed)),
    }


def evaluate(model, X, y, threshold=0.5) -> dict:
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y, p),
        "Accuracy": accuracy_score(y, yhat),
        "Recall (sensitivity)": recall_score(y, yhat),
        "Specificity": recall_score(1 - y, 1 - yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat),
        "Brier": brier_score_loss(y, p),
    }


def subgroup_report(model, X, y, feature: str, threshold=0.5, min_n=50) -> pd.DataFrame:
    """Performance per value of a (sensitive) feature -> bias / error analysis."""
    rows = []
    p = model.predict_proba(X)[:, 1]
    for v in sorted(X[feature].unique()):
        m = (X[feature] == v).to_numpy()
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        yhat = (p[m] >= threshold).astype(int)
        rows.append({
            "group": describe_value(feature, v), "n": int(m.sum()),
            "prevalence": y[m].mean(), "mean predicted risk": p[m].mean(),
            "AUC": roc_auc_score(y[m], p[m]),
            "Recall": recall_score(y[m], yhat), "Specificity": recall_score(1 - y[m], 1 - yhat),
            "FNR (missed cases)": 1 - recall_score(y[m], yhat),
        })
    return pd.DataFrame(rows).set_index("group")


def error_cases(model, X, y, threshold=0.5, k=10):
    """Most confident false negatives / false positives - edge cases for the audit."""
    p = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="risk")
    df = X.assign(risk=p, y=y)
    fn = df[(df.y == 1) & (df.risk < threshold)].nsmallest(k, "risk")
    fp = df[(df.y == 0) & (df.risk >= threshold)].nlargest(k, "risk")
    return fn, fp


# =============================================================================
# 3. XAI METHODS (common interface: everything returns a DataFrame / Series
#    indexed by feature, in probability units)
# =============================================================================
def proba_fn(model):
    return lambda X: model.predict_proba(pd.DataFrame(np.asarray(X), columns=FEATURES))[:, 1]


def perm_importance(model, X, y, n_repeats=10, scoring="roc_auc", seed=SEED) -> pd.DataFrame:
    """Global, model-agnostic: drop in AUC when a feature is shuffled."""
    r = permutation_importance(model, X, y, n_repeats=n_repeats, scoring=scoring, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std}, index=X.columns)
              .sort_values("mean", ascending=False))


class ShapExplainer:
    """SHAP in probability space for any of our models.
    - Tree models -> TreeExplainer (exact, fast).
    - Anything else -> model-agnostic PermutationExplainer.
    """

    def __init__(self, model, X_background: pd.DataFrame, n_background=200, seed=SEED):
        import shap
        self.model = model
        bg = X_background.sample(min(n_background, len(X_background)), random_state=seed)
        if isinstance(model, HistGradientBoostingClassifier):
            self._exp = shap.TreeExplainer(model, data=bg, model_output="probability")
        elif isinstance(model, RandomForestClassifier):
            # Path-dependent TreeSHAP; RF "raw" output is already a probability.
            self._exp = shap.TreeExplainer(model)
        else:
            masker = shap.maskers.Independent(bg, max_samples=n_background)
            self._exp = shap.PermutationExplainer(proba_fn(model), masker)

    def __call__(self, X: pd.DataFrame):
        """Returns a shap.Explanation (n_samples x n_features) for the positive class."""
        import shap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = self._exp(X, silent=True) if isinstance(self._exp, shap.PermutationExplainer) else self._exp(X)
        if ex.values.ndim == 3:  # (n, features, classes) -> class 1
            ex = ex[:, :, 1]
        ex.data = X.to_numpy()
        ex.feature_names = list(X.columns)
        return ex

    @staticmethod
    def global_importance(ex) -> pd.Series:
        return pd.Series(np.abs(ex.values).mean(0), index=ex.feature_names).sort_values(ascending=False)


class LimeExplainer:
    """LIME tabular explainer; local linear surrogate around one person."""

    def __init__(self, model, X_train: pd.DataFrame, seed=SEED):
        from lime.lime_tabular import LimeTabularExplainer
        self.model = model
        self._exp = LimeTabularExplainer(
            X_train.to_numpy(), feature_names=FEATURES, class_names=["No diabetes", "(Pre)diabetes"],
            categorical_features=[FEATURES.index(f) for f in CATEGORICAL if CODEBOOK[f].kind == "binary"],
            discretize_continuous=True, mode="classification", random_state=seed,
        )

    def explain(self, x: pd.Series, num_features=len(FEATURES), num_samples=5000):
        """Returns (weights Series indexed by feature, raw LIME explanation)."""
        e = self._exp.explain_instance(x.to_numpy(), lambda Z: self.model.predict_proba(
            pd.DataFrame(Z, columns=FEATURES)), num_features=num_features, num_samples=num_samples)
        w = pd.Series({FEATURES[i]: v for i, v in e.as_map()[1]})
        return w.reindex(FEATURES).fillna(0.0), e


# =============================================================================
# 4. VALIDATING THE EXPLANATIONS
# =============================================================================
def rank_agreement(importances: dict[str, pd.Series]) -> pd.DataFrame:
    """Spearman rank correlation between global importance vectors
    (e.g. permutation vs mean|SHAP|, or model A vs model B)."""
    names = list(importances)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            s1, s2 = importances[a].reindex(FEATURES), importances[b].reindex(FEATURES)
            M.loc[a, b] = spearmanr(s1, s2).statistic
    return M


def topk_overlap(a: pd.Series, b: pd.Series, k=5) -> float:
    """Jaccard overlap of the k features with largest |attribution| (local agreement)."""
    ta, tb = set(a.abs().nlargest(k).index), set(b.abs().nlargest(k).index)
    return len(ta & tb) / len(ta | tb)


def deletion_test(model, X: pd.DataFrame, attributions: np.ndarray, X_reference: pd.DataFrame,
                  ks=(0, 1, 2, 3, 5, 8, 13, 21), n_random=5, seed=SEED) -> pd.DataFrame:
    """Faithfulness: replace the top-k features (by |attribution|) with values from a
    random reference person and track how much the prediction changes. A faithful
    explanation changes the prediction faster than removing k random features."""
    rng = np.random.default_rng(seed)
    f = proba_fn(model)
    X_np = X.to_numpy().astype(float)
    ref = X_reference.sample(len(X), replace=True, random_state=seed).to_numpy().astype(float)
    p0 = f(X_np)
    order = np.argsort(-np.abs(attributions), axis=1)
    rows = []
    for k in ks:
        Xe = X_np.copy()
        idx = order[:, :k]
        np.put_along_axis(Xe, idx, np.take_along_axis(ref, idx, 1), 1)
        expl = np.abs(f(Xe) - p0).mean()
        rnd = []
        for _ in range(n_random):
            ridx = np.argsort(rng.random(X_np.shape), axis=1)[:, :k]
            Xr = X_np.copy()
            np.put_along_axis(Xr, ridx, np.take_along_axis(ref, ridx, 1), 1)
            rnd.append(np.abs(f(Xr) - p0).mean())
        rows.append({"k": k, "explanation": expl, "random": np.mean(rnd)})
    return pd.DataFrame(rows).set_index("k")


def lime_stability(lime: LimeExplainer, x: pd.Series, n_runs=5, k=5) -> dict:
    """Run LIME repeatedly with different seeds -> how stable is the top-k?"""
    from lime.lime_tabular import LimeTabularExplainer  # noqa: F401  (ensures lime installed)
    runs = []
    base_state = lime._exp.random_state
    for i in range(n_runs):
        lime._exp.random_state = np.random.RandomState(SEED + i)
        runs.append(lime.explain(x)[0])
    lime._exp.random_state = base_state
    pairs = [topk_overlap(runs[i], runs[j], k) for i in range(n_runs) for j in range(i + 1, n_runs)]
    W = pd.DataFrame(runs)
    return {"mean_topk_jaccard": float(np.mean(pairs)), "weight_std": W.std().mean(), "runs": W}


def advice_safety(model, X: pd.DataFrame, features=None) -> pd.DataFrame:
    """For each actionable feature, apply the *healthy* change to everyone for whom it
    is possible and measure the change in predicted risk. A healthy change that
    INCREASES predicted risk is an unsafe signal: explanations built on this model
    would tell people that an unhealthy behaviour protects them."""
    f = proba_fn(model)
    rows = []
    for feat in features or ACTIONABLE + MEDICAL:
        d = HEALTHY_DIRECTION[feat]
        Xc = X.copy()
        if CODEBOOK[feat].kind == "binary":
            mask = (Xc[feat] == (0 if d > 0 else 1)).to_numpy()
            Xc.loc[mask, feat] = 1 if d > 0 else 0
            change = "0 -> 1" if d > 0 else "1 -> 0"
        elif feat == "BMI":
            mask = (Xc[feat] > 25).to_numpy()
            Xc.loc[mask, feat] = (Xc.loc[mask, feat] - 3).clip(lower=18)
            change = "-3 BMI points (if BMI>25)"
        else:
            step = d * (1 if CODEBOOK[feat].kind == "ordinal" else 5)
            lo, hi = X[feat].min(), X[feat].max()
            mask = (Xc[feat] != (lo if d < 0 else hi)).to_numpy()
            Xc.loc[mask, feat] = (Xc.loc[mask, feat] + step).clip(lo, hi)
            change = f"{step:+d} ({'level' if CODEBOOK[feat].kind == 'ordinal' else 'days'})"
        if mask.sum() == 0:
            continue
        delta = f(Xc[mask]) - f(X[mask])
        rows.append({"feature": feat, "healthy change": change, "n affected": int(mask.sum()),
                     "mean Δ risk (pp)": 100 * delta.mean(),
                     "% where risk goes UP": 100 * (delta > 0.005).mean()})
    out = pd.DataFrame(rows).set_index("feature")
    out["flag"] = np.where(out["mean Δ risk (pp)"] > 0, "UNSAFE", "ok")
    return out


# =============================================================================
# 5. ACTIONABILITY: what-if and counterfactuals
# =============================================================================
def what_if(model, x: pd.Series, **changes) -> dict:
    """what_if(model, x, BMI=27, PhysActivity=1) -> risk before/after."""
    f = proba_fn(model)
    x2 = x.copy()
    for k, v in changes.items():
        x2[k] = v
    return {"before": float(f(x.to_frame().T)[0]), "after": float(f(x2.to_frame().T)[0]), "x_new": x2}


def action_candidates(x: pd.Series, bmi_steps=(1, 2, 3, 5), bmi_floor=22) -> list[dict]:
    """Realistic single changes a person could make (only healthy directions)."""
    cands = []
    for s in bmi_steps:
        if x["BMI"] - s >= bmi_floor:
            cands.append({"BMI": x["BMI"] - s})
    for feat in ["PhysActivity", "Fruits", "Veggies"]:
        if x[feat] == 0:
            cands.append({feat: 1})
    if x["HvyAlcoholConsump"] == 1:
        cands.append({"HvyAlcoholConsump": 0})
    return cands


def counterfactual_search(model, x: pd.Series, max_changes=3, target=None) -> pd.DataFrame:
    """Greedy search over *healthy, actionable* changes only. At every step pick the
    change that lowers risk most; stop at max_changes or when risk < target."""
    f = proba_fn(model)
    cur, used, path = x.copy(), set(), []
    p_cur = float(f(cur.to_frame().T)[0])
    path.append({"step": "current", "risk": p_cur})
    for _ in range(max_changes):
        best = None
        for c in action_candidates(cur):
            (feat, val), = c.items()
            if feat in used and feat != "BMI":
                continue
            xc = cur.copy(); xc[feat] = val
            p = float(f(xc.to_frame().T)[0])
            if best is None or p < best[2]:
                best = (feat, val, p)
        if best is None or best[2] >= p_cur - 1e-4:
            break
        feat, val, p_cur = best
        path.append({"step": f"{label(feat)}: {describe_value(feat, cur[feat])} -> {describe_value(feat, val)}",
                     "risk": p_cur})
        cur[feat] = val; used.add(feat)
        if target is not None and p_cur < target:
            break
    return pd.DataFrame(path)


# =============================================================================
# 6. PLOT STYLE (shared by all stakeholder figures)
# =============================================================================
COLORS = {
    "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df",
    "increase": "#e34948",   # pushes risk UP   (diverging red pole)
    "decrease": "#2a78d6",   # pushes risk DOWN (diverging blue pole)
    "neutral": "#b4b2ab",
}
MODEL_COLORS = {"Logistic regression": "#2a78d6", "Random forest": "#eb6834", "Gradient boosting": "#1baf7a",
                "Neural network": "#eda100"}


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": COLORS["surface"], "axes.facecolor": COLORS["surface"],
        "savefig.facecolor": COLORS["surface"], "figure.dpi": 110, "savefig.dpi": 200,
        "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["text2"],
        "xtick.color": COLORS["text2"], "ytick.color": COLORS["text2"], "text.color": COLORS["text"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def savefig(fig, stakeholder: str, name: str):
    """figures/<stakeholder>/<name>.png - one folder per audience."""
    d = FIG_DIR / stakeholder
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", bbox_inches="tight")


def plot_contributions(ax, contrib: pd.Series, title="", xlabel="Change in predicted risk (percentage points)",
                       plain_labels=True, max_features=10):
    """Horizontal diverging bars: red = raises risk, blue = lowers risk."""
    c = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:max_features][::-1] * 100
    names = [label(f) if plain_labels else f for f in c.index]
    ax.barh(names, c.values, color=[COLORS["increase"] if v > 0 else COLORS["decrease"] for v in c.values],
            height=0.6, edgecolor=COLORS["surface"], linewidth=2)
    ax.axvline(0, color=COLORS["text2"], lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    return ax
