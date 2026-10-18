"""
Framework for Assignment 1: Auditing and Explaining Diabetes Risk Prediction Models.

The notebook (projekt.ipynb) tells the story; this module holds the reusable parts:

    1. Data      - loading, codebook (labels, actionability, sensitivity), splitting
    2. Models    - decision tree + neural network, tuning sweeps, evaluation (overall and per subgroup)
    3. XAI       - SHAP and LIME behind one common interface
    4. Validation- SHAP/LIME agreement, recovery of the tree's true features, deletion
                   (faithfulness) test, LIME stability, "safety" check of actionable advice
    5. Actions   - what-if / counterfactual search over actionable features only
    6. Plot style- shared colours, matplotlib style and the stakeholder figure helpers

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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
    actionable: str         # "yes" (a habit the person can change) | "medical" (health status: not one
                            # action, but can be managed with a doctor) | "no"
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
def make_tree(max_depth=5, min_samples_leaf=100, seed=SEED):
    """Glass box: a shallow tree is its own explanation (the rules can be read off)."""
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)


def make_nn(hidden=(64, 32), alpha=1e-3, seed=SEED):
    """Black box: smooth non-linear function with interactions; needs scaled inputs
    and can only be explained post hoc."""
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=hidden, activation="relu", alpha=alpha, learning_rate_init=1e-3,
        batch_size=256, max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=seed))


def make_models(tree_params: dict | None = None, nn_params: dict | None = None) -> dict:
    """The two candidate models, as different as possible in inductive bias and
    interpretability: transparent decision tree vs black-box neural network."""
    return {"Decision tree": make_tree(**(tree_params or {})),
            "Neural network": make_nn(**(nn_params or {}))}


def tune_tree(X_train, y_train, X_val, y_val, depths=(3, 4, 5, 6, 8, 10, None)) -> pd.DataFrame:
    """Validation AUC vs depth. Deeper trees gain little AUC but quickly become unreadable."""
    rows = []
    for d in depths:
        t = make_tree(max_depth=d).fit(X_train, y_train)
        rows.append({"max_depth": d if d else "none", "leaves": t.get_n_leaves(),
                     "features used": len(tree_features(t)), "AUC": evaluate(t, X_val, y_val)["AUC"]})
    return pd.DataFrame(rows).set_index("max_depth")


def tune_nn(X_train, y_train, X_val, y_val, hiddens=((32,), (64, 32), (128, 64)), alphas=(1e-4, 1e-3, 1e-2)) -> pd.DataFrame:
    """Small grid over network size and L2 penalty, scored on the validation set."""
    rows = []
    for h in hiddens:
        for a in alphas:
            m = make_nn(hidden=h, alpha=a).fit(X_train, y_train)
            rows.append({"hidden": h, "alpha": a, "parameters": sum(w.size for w in m[-1].coefs_),
                         "AUC": evaluate(m, X_val, y_val)["AUC"]})
    return pd.DataFrame(rows)


def model_columns(model) -> list[str]:
    """Features a fitted model was trained on (all of FEATURES, or a subset in the bias audit)."""
    return list(getattr(model, "feature_names_in_", FEATURES))


def tree_features(tree) -> list[str]:
    """Features the fitted tree actually splits on - the ground truth for checking explanations."""
    cols = model_columns(tree)
    used = {cols[i] for i in tree.tree_.feature if i >= 0}
    return [f for f in cols if f in used]


def evaluate(model, X, y, threshold=0.5) -> dict:
    p = proba_fn(model)(X)
    yhat = (p >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y, p),
        "Accuracy": accuracy_score(y, yhat),
        "Recall (sensitivity)": recall_score(y, yhat),
        "Specificity": recall_score(1 - y, 1 - yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat),
    }


def subgroup_report(model, X, y, feature: str, threshold=0.5, min_n=50) -> pd.DataFrame:
    """Performance per value of a (sensitive) feature -> bias / error analysis."""
    rows = []
    p = proba_fn(model)(X)  # X may hold more columns than the model uses (grouping feature)
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
    """Risk as a function of a DataFrame (extra columns are dropped) or of a plain array."""
    cols = model_columns(model)

    def f(X):
        X = X[cols] if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=cols)
        return model.predict_proba(X)[:, 1]
    return f


class ShapExplainer:
    """SHAP in probability space, with the same background sample for both models
    (interventional SHAP), so attributions are directly comparable.
    - Decision tree  -> TreeExplainer (exact, fast).
    - Neural network -> model-agnostic PermutationExplainer.
    """

    def __init__(self, model, X_background: pd.DataFrame, n_background=200, seed=SEED):
        import shap
        self.model = model
        bg = X_background.sample(min(n_background, len(X_background)), random_state=seed)
        if isinstance(model, DecisionTreeClassifier):
            # A classification tree's raw output already is the class probability.
            self._exp = shap.TreeExplainer(model, data=bg)
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

    def explain_many(self, X: pd.DataFrame, num_samples=2000):
        """LIME for every row of X -> (weights DataFrame like X, surrogate R^2 per row)."""
        W, r2 = [], []
        for _, xi in X.iterrows():
            w, e = self.explain(xi, num_samples=num_samples)
            W.append(w); r2.append(e.score)
        return pd.DataFrame(W, index=X.index), pd.Series(r2, index=X.index, name="LIME R^2")


# =============================================================================
# 4. VALIDATING THE EXPLANATIONS
# =============================================================================
def topk_overlap(a: pd.Series, b: pd.Series, k=5) -> float:
    """Jaccard overlap of the k features with largest |attribution| (local agreement)."""
    ta, tb = set(a.abs().nlargest(k).index), set(b.abs().nlargest(k).index)
    return len(ta & tb) / len(ta | tb)


def topk_share(a: pd.Series, features, k=5) -> float:
    """Share of the k largest |attributions| that fall on `features` (e.g. the tree's true features)."""
    return len(set(a.abs().nlargest(k).index) & set(features)) / k


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


def counterfactual_search(model, x: pd.Series, max_changes=3, target=None, exclude=()) -> pd.DataFrame:
    """Greedy search over *healthy, actionable* changes only. At every step pick the
    change that lowers risk most; stop at max_changes or when risk < target.
    Features in `exclude` (flagged unsafe by advice_safety) are never suggested: even if
    the change happens to help this person, the model is not trustworthy on that feature."""
    f = proba_fn(model)
    cur, used, path = x.copy(), set(), []
    p_cur = float(f(cur.to_frame().T)[0])
    path.append({"step": "current", "risk": p_cur})
    for _ in range(max_changes):
        best = None
        for c in action_candidates(cur):
            (feat, val), = c.items()
            if feat in exclude or (feat in used and feat != "BMI"):
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
# 6. PLOT STYLE + STAKEHOLDER FIGURES
# =============================================================================
COLORS = {
    "surface": "white", "text": "black", "text2": "dimgray", "grid": "lightgray",
    "increase": "tab:red",    # pushes risk UP
    "decrease": "tab:blue",   # pushes risk DOWN
    "neutral": "gray",
}
# Model colours avoid red/blue, which are used for "raises/lowers risk".
MODEL_COLORS = {"Decision tree": "tab:green", "Neural network": "tab:orange"}


def set_style():
    plt.style.use("default")
    mpl.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})


def savefig(fig, stakeholder: str, name: str):
    """figures/<stakeholder>/<name>.png - one folder per audience."""
    d = FIG_DIR / stakeholder
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", bbox_inches="tight")


def plot_contributions(ax, contrib: pd.Series, title="", xlabel="Change in predicted risk (percentage points)",
                       plain_labels=True, max_features=10, min_abs=0.0, values: pd.Series | None = None):
    """Horizontal diverging bars: red = raises risk, blue = lowers risk.
    `min_abs` hides features with (near) zero contribution, e.g. features a tree never uses.
    `values` (one person's feature values) adds the value to each label, e.g. "Physically active: No" -
    without it a red "Physically active" bar reads as "being active raises your risk"."""
    c = contrib[contrib.abs() > min_abs]
    c = c.reindex(c.abs().sort_values(ascending=False).index)[:max_features][::-1] * 100
    names = [label(f) if plain_labels else f for f in c.index]
    if values is not None:
        names = [f"{n}: {describe_value(f, values[f])}" if f in CODEBOOK else n for n, f in zip(names, c.index)]
    ax.barh(names, c.values, color=[COLORS["increase"] if v > 0 else COLORS["decrease"] for v in c.values], height=0.6)
    ax.axvline(0, color=COLORS["text2"], lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    return ax


def plot_grouped_bars(ax, values: pd.DataFrame, title="", xlabel="", colors=None):
    """One horizontal bar per model (columns) for every feature (rows); colours from MODEL_COLORS
    unless `colors` (column -> colour) is given."""
    colors = {**MODEL_COLORS, **(colors or {})}
    n = values.shape[1]
    h = 0.8 / n
    y = np.arange(len(values))
    for i, name in enumerate(values.columns):
        ax.barh(y + (i - (n - 1) / 2) * h, values[name], height=h, color=colors.get(name), label=name)
    ax.set_yticks(y, values.index)
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel=xlabel)
    ax.legend()
    return ax


def _plain_question(feature: str, threshold: float) -> tuple[str, str]:
    """A split 'feature <= threshold' as a yes/no question, plus the answer that leads to the LEFT
    child (feature <= threshold). Yes/no features are asked the positive way round
    ('High blood pressure?'), so for them the left child is 'no' - no double negatives."""
    kind, values = CODEBOOK[feature].kind, CODEBOOK[feature].values
    if kind == "binary":
        return f"{label(feature)}?", "no"
    if feature == "Age":
        return f"Younger than {AGE_BANDS[int(threshold) + 1].split('-')[0].rstrip('+')}?", "yes"
    if feature == "GenHlth":
        return f"General health {values[int(threshold)].lower()} or better?", "yes"
    if kind == "ordinal" and values:
        below = [v for k, v in values.items() if k <= threshold]
        return f"{label(feature)}: {' or '.join(below) if len(below) <= 2 else 'up to ' + below[-1]}?", "yes"
    name = "BMI" if feature == "BMI" else label(feature)
    return f"{name} {int(threshold)} or lower?", "yes"


def plot_tree_logic(ax, tree, max_depth=3, fontsize=8):
    """The decision tree as a plain-language flowchart: every box asks a question, every arrow
    says which answer it stands for, and every box shows how many people end up there and how
    many of them have (pre)diabetes. The bottom row is shown as risk groups (the questions below
    it are cut off). Box colour follows the red/blue risk convention."""
    tint = mpl.colors.LinearSegmentedColormap.from_list("risk", ["#a9c9f0", "#ffffff", "#f2a4a3"])
    nodes = plot_tree(tree, max_depth=max_depth, feature_names=model_columns(tree), impurity=False,
                      proportion=True, rounded=True, precision=2, fontsize=fontsize, ax=ax)
    visible = []  # (annotation, question, answer of the left child, share, risk)
    for node in nodes:
        lines = node.get_text().split("\n")
        value = next((l for l in lines if l.startswith("value")), None)
        if value is None:            # "(...)" placeholder for a cut-off subtree
            node.set_visible(False)
            continue
        risk = float(value.split("[")[1].rstrip("]").split(",")[1])
        share = float(next(l for l in lines if l.startswith("samples")).split("= ")[1].rstrip("%"))
        question, left = None, None
        if " <= " in lines[0]:
            feat, thr = lines[0].split(" <= ")
            question, left = _plain_question(feat, float(thr))
        visible.append((node, question, left, share, risk))

    # Stretch the visible levels over the full height (the hidden placeholder row would leave a gap).
    # Positions are in axes fractions: xyann = the box, xy = where its arrow starts (the parent box).
    ys = [n.xyann[1] for n, *_ in visible]
    lo, hi = min(ys), max(ys)
    stretch = lambda y: 0.05 + 0.9 * (y - lo) / (hi - lo) if hi > lo else y
    for node, question, left, share, risk in visible:
        text = [f"{share:.0f}% of people", f"{risk:.0%} have (pre)diabetes"]
        if question and node.xyann[1] > lo:   # bottom row = risk groups, their questions are cut off
            text.insert(0, question)
        node.set_text("\n".join(text))
        node.get_bbox_patch().set_facecolor(tint(risk))
        node.xyann = (node.xyann[0], stretch(node.xyann[1]))
        node.xy = (node.xy[0], stretch(node.xy[1]))

    # Label every arrow with its answer: find the parent (whose box sits where the arrow starts).
    boxes = {(round(n.xyann[0], 6), round(n.xyann[1], 6)): (n, left) for n, _, left, _, _ in visible}
    for node, *_ in visible:
        parent = boxes.get((round(node.xy[0], 6), round(node.xy[1], 6)))
        if parent is None or parent[0] is node:   # the root has no arrow
            continue
        left = parent[1]
        answer = left if node.xyann[0] < parent[0].xyann[0] else {"yes": "no", "no": "yes"}[left]
        ax.text((node.xy[0] + node.xyann[0]) / 2, (node.xy[1] + node.xyann[1]) / 2, answer,
                transform=ax.transAxes, ha="center", va="center", fontsize=fontsize + 1, fontweight="bold",
                color=COLORS["text2"], bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["surface"], ec="none"))
    ax.grid(False)
    return ax


# Not actionable and socio-economic: shown to the patient as one summed bar (SHAP values are additive).
BACKGROUND = ["Sex", "Income", "Education"]
BACKGROUND_LABEL = "Background (sex, income, education)"


def patient_figure(contrib: pd.Series, x: pd.Series, cf: pd.DataFrame, risk: float, average: float, hide=(), subtitle=""):
    """Three panels for the patient: what raises/lowers *their* score (things they can change
    vs everything else) and the effect of realistic changes.

    Design choices (good explanations are selected, contrastive and truthful):
    - a score out of 100 compared with the average person (the SHAP base value), not a
      "% chance": the training data is 50/50, so the raw probability is no real-world chance;
    - only habits count as "things you can change". High blood pressure / cholesterol are
      "ever told you have ..." answers - they belong with the doctor, not in a to-do list;
    - sex, income and education are summed into one background bar: not actionable, and
      "your income raises your risk" can stigmatise without helping;
    - features in `hide` (flagged unsafe by advice_safety) are never shown."""
    habits = contrib[[f for f in ACTIONABLE if f not in hide]]
    other = contrib[[f for f in FEATURES if CODEBOOK[f].actionable != "yes" and f not in BACKGROUND + list(hide)]].copy()
    other[BACKGROUND_LABEL] = contrib[BACKGROUND].sum()
    score, avg = round(100 * risk), round(100 * average)

    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.75)
    ax = plot_contributions(fig.add_subplot(gs[0]), habits, title="Things you can change yourself",
                            xlabel="Effect on your score (points)", max_features=5, min_abs=1e-3, values=x)
    if (habits.abs() <= 1e-3).all():
        ax.text(0.5, 0.5, "None of your habits change\nthis model's score", transform=ax.transAxes,
                ha="center", va="center", color=COLORS["text2"])
    plot_contributions(fig.add_subplot(gs[1]), other, title="Your health and background",
                       xlabel="Effect on your score (points)", max_features=5, min_abs=1e-3, values=x)

    ax = fig.add_subplot(gs[2])
    steps = cf.copy(); steps["risk"] *= 100
    ax.bar(range(len(steps)), steps["risk"], color=[COLORS["increase"]] + [COLORS["decrease"]] * (len(steps) - 1), width=0.6)
    for i, r in enumerate(steps["risk"]):
        ax.text(i, r + 1.5, f"{r:.0f}", ha="center", fontweight="bold")
    ax.axhline(avg, color=COLORS["text2"], lw=1, ls="--", label=f"Average person ({avg})")
    ax.legend(loc="upper right", fontsize=9)
    short = (s.replace("Body mass index (BMI)", "BMI").replace("Physically active", "Exercise")
              .replace("Eats vegetables daily", "Vegetables").replace("Eats fruit daily", "Fruit")
              .replace(" -> ", "→").replace(": ", "\n") for s in steps["step"][1:])
    ax.set_xticks(range(len(steps)), ["Today"] + list(short))
    ax.set(ylim=(0, 100), title="If you change your habits...", ylabel="Risk score")
    ax.grid(axis="x", visible=False)
    if len(steps) == 1:
        ax.text(0.5, 0.8, "This model finds no realistic\nchange that lowers your score", transform=ax.transAxes,
                ha="center", va="top", color=COLORS["text2"])

    fig.suptitle(f"Your diabetes risk score: {score} out of 100", fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.06)
    compare = "higher than" if score > avg else "lower than" if score < avg else "the same as"
    fig.text(0.01, 0.97, f"This is {compare} the average person in our data (score {avg}). {subtitle}",
             ha="left", color=COLORS["text2"])
    note = ("How to read this: red bars raise your score, blue bars lower it. The score shows how much your answers resemble those of "
            "people in our data who have\n(pre)diabetes - half of them did - so it is not your chance of getting diabetes, and it is not a "
            "diagnosis. The bars show what the model reacts to,\nnot proven causes. "
            "Talk to your doctor about blood pressure, cholesterol and your general health.")
    if hide:
        note += "\nSome habits are left out because the model is not reliable about them."
    fig.text(0.01, -0.03, note, ha="left", va="top", color=COLORS["text2"], fontsize=9)
    return fig
