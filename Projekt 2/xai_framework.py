"""
Framework for Project 2: Auditing and Explaining Diabetes Risk Prediction Models on the imbalanced,
three-class BRFSS 2015 data. Built on the Project 1 framework.

The notebook (projekt2.ipynb) tells the story; this module holds the reusable parts:

    1. Data      - loading, codebook (labels, actionability, sensitivity), splitting
    2. Models    - logistic regression + neural network, tuning sweeps (incl. depth), threshold choice, evaluation
                   (overall and per subgroup), the logistic regression's exact contributions and points table
    3. XAI       - SHAP and LIME behind one common interface
    4. Validation- SHAP/LIME agreement, recovery of the logistic regression's exact contributions, deletion
                   (faithfulness) test, LIME stability, "safety" check of actionable advice
    5. Actions   - what-if / counterfactual search over actionable features only
    6. Audit     - what the network has learned: hidden representations and 2-D embeddings of them
                   (PCA, random projection, t-SNE, UMAP), concept probes with a random-label baseline,
                   five ways to build a CAV, TCAV, removing a concept inside the network (ablation),
                   and NMF concept discovery
    7. Plot style- one colour system for every figure and the stakeholder figure helpers

Changes from Project 1:
    - the target is Diabetes_012 (0 = none, 1 = prediabetes, 2 = diabetes). The models predict
      (pre)diabetes (classes 1 and 2 merged), as in Project 1; the three-class label is kept for
      stratification and for analyses per class
    - the data is imbalanced, so the decision threshold is chosen on the validation set instead of 0.5,
      and the evaluation adds balanced accuracy, average precision and calibration
    - the colour system from the redesigned Project 1 figures (week 8)

All SHAP / LIME attributions are expressed in *probability* units (percentage
points of predicted risk), so they are comparable across models and easy to
translate for non-technical stakeholders.
"""
from __future__ import annotations

import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET_3 = "Diabetes_012"          # the label in the file: 0 none, 1 prediabetes, 2 diabetes
TARGET = "Diabetes_binary"         # what the models predict: 1 = prediabetes or diabetes
CLASS_NAMES = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}
DATA_PATH = Path(__file__).parent / "diabetes_012_health_indicators_BRFSS2015.csv"
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
    """The three-class file plus the merged binary target the models predict."""
    df = pd.read_csv(path).astype(int)
    assert list(df.columns) == [TARGET_3] + FEATURES, "Unexpected columns"
    df[TARGET] = (df[TARGET_3] > 0).astype(int)
    return df


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1, seed=SEED, target=TARGET):
    """Train / validation / test split, stratified on the THREE-class label so the small prediabetes
    class has the same share in every part. Validation is for model selection, tuning and the
    threshold; the test set is only touched for final numbers and explanations."""
    X, y, strata = df[FEATURES], df[target], df[TARGET_3]
    X_tmp, X_test, y_tmp, y_test, s_tmp, _ = train_test_split(X, y, strata, test_size=test_size,
                                                              stratify=strata, random_state=seed)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=rel_val, stratify=s_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 2. MODELS + EVALUATION
# =============================================================================
def make_lr(C=1.0, class_weight=None):
    """Glass box: one weight per feature in log-odds, the same for everyone - the weights are
    the explanation. Inputs are standardised so the L2 penalty treats all features alike."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=class_weight, max_iter=2000))


def make_nn(hidden=(64, 32), alpha=1e-3, seed=SEED):
    """Black box: smooth non-linear function with interactions; needs scaled inputs
    and can only be explained post hoc."""
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=hidden, activation="relu", alpha=alpha, learning_rate_init=1e-3,
        batch_size=256, max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=seed))


def make_models(lr_params: dict | None = None, nn_params: dict | None = None) -> dict:
    """The two models: transparent logistic regression (additive, no interactions) vs
    black-box neural network (non-linear, with interactions)."""
    return {"Logistic regression": make_lr(**(lr_params or {})),
            "Neural network": make_nn(**(nn_params or {}))}


def tune_lr(X_train, y_train, X_val, y_val, Cs=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10)) -> pd.DataFrame:
    """Validation AUC vs the inverse L2 penalty C."""
    return pd.DataFrame([{"C": C, "AUC": evaluate(make_lr(C).fit(X_train, y_train), X_val, y_val)["AUC"]}
                         for C in Cs]).set_index("C")


def tune_nn(X_train, y_train, X_val, y_val, hiddens=((32,), (64, 32), (128, 64)), alphas=(1e-4, 1e-3, 1e-2)) -> pd.DataFrame:
    """Small grid over network size and L2 penalty, scored on the validation set."""
    rows = []
    for h in hiddens:
        for a in alphas:
            m = make_nn(hidden=h, alpha=a).fit(X_train, y_train)
            rows.append({"hidden": h, "alpha": a, "parameters": sum(w.size for w in m[-1].coefs_),
                         "AUC": evaluate(m, X_val, y_val)["AUC"]})
    return pd.DataFrame(rows)


def compare_depth(X_train, y_train, X_val, y_val, hiddens, alpha=1e-2, seeds=(SEED, SEED + 1, SEED + 2)) -> pd.DataFrame:
    """Validation AUC of networks with different hidden layers, each trained with several seeds so a
    difference can be compared with the spread from the seed alone. Also counts dead units (never active
    on 4000 validation persons) in the last hidden layer, which shows when a layer is too narrow."""
    rows = []
    for h in hiddens:
        aucs, dead = [], []
        for s in seeds:
            m = make_nn(hidden=h, alpha=alpha, seed=s).fit(X_train, y_train)
            aucs.append(evaluate(m, X_val, y_val)["AUC"])
            dead.append(int((hidden_activations(m, X_val.iloc[:4000], len(h) - 1).max(0) == 0).sum()))
        rows.append({"hidden layers": " + ".join(map(str, h)), "depth": len(h),
                     "parameters": sum(w.size for w in m[-1].coefs_), "AUC (mean)": np.mean(aucs),
                     "AUC (sd over seeds)": np.std(aucs), "dead units in last layer (mean)": np.mean(dead),
                     "units in last layer": h[-1]})
    return pd.DataFrame(rows).set_index("hidden layers")


THRESHOLD_RULES = {"Youden's J": dict(rule="youden"), "Recall 80 %": dict(rule="recall", target=0.8)}


def choose_threshold(model, X_val, y_val, rule="youden", target=0.8) -> float:
    """Decision threshold chosen on the validation set (with 16 % positives, 0.5 flags almost nobody).
    rule="youden": maximise recall + specificity - 1, i.e. weigh a missed case and a false alarm by their rates.
    rule="recall": the highest threshold that still catches `target` of the persons with (pre)diabetes; the
    share to catch is a decision for the stakeholders, and the false alarms follow from it."""
    fpr, tpr, thr = roc_curve(y_val, proba_fn(model)(X_val))
    if rule == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    if rule == "recall":
        return float(thr[np.argmax(tpr >= target)])
    raise ValueError(rule)


def model_columns(model) -> list[str]:
    """Features a fitted model was trained on (all of FEATURES, or a subset in the bias audit)."""
    return list(getattr(model, "feature_names_in_", FEATURES))


def lr_weights(model) -> pd.Series:
    """Logistic regression weights per ORIGINAL unit (log-odds per BMI point, per yes, ...)."""
    scaler, lr = model[0], model[-1]
    return pd.Series(lr.coef_[0] / scaler.scale_, index=model_columns(model))


def lr_contributions(model, X: pd.DataFrame, X_background: pd.DataFrame, n_background=200, seed=SEED) -> pd.DataFrame:
    """The logistic regression's EXACT contribution of every feature for every person, in
    log-odds: weight x (person's value - background mean). The same background sample as
    ShapExplainer, so this is the ground truth SHAP and LIME should recover (direction, ranking)."""
    bg = X_background.sample(min(n_background, len(X_background)), random_state=seed)
    cols = model_columns(model)
    return (X[cols] - bg[cols].mean()) * lr_weights(model)


# Units for the director's points table: a point = 0.1 log-odds per unit below (default: per 1).
POINT_UNITS = {"BMI": 5, "MentHlth": 10, "PhysHlth": 10}
POINT_LABELS = {
    "BMI": "BMI (per 5 points higher)", "Age": "Age (per 5 years older)",
    "GenHlth": "General health (per step worse)", "Income": "Income (per step higher)",
    "Education": "Education (per step higher)", "MentHlth": "Poor mental health (per 10 days)",
    "PhysHlth": "Poor physical health (per 10 days)", "Sex": "Male",
}


def lr_points(model, point=0.1) -> pd.Series:
    """The logistic regression as a points table: whole points per answer, adding up to the score.
    Indexed by feature name; use point_label() for the text shown to the director."""
    w = lr_weights(model)
    pts = pd.Series({f: round(w[f] * POINT_UNITS.get(f, 1) / point) for f in w.index})
    return pts[pts != 0]


def point_label(feature: str) -> str:
    return POINT_LABELS.get(feature, label(feature))


def evaluate(model, X, y, threshold=0.5) -> dict:
    p = proba_fn(model)(X)
    yhat = (p >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y, p),
        "Average precision": average_precision_score(y, p),
        "Accuracy": accuracy_score(y, yhat),
        "Balanced accuracy": balanced_accuracy_score(y, yhat),
        "Recall (sensitivity)": recall_score(y, yhat),
        "Specificity": recall_score(1 - y, 1 - yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat),
    }


def recall_per_class(model, X, y3, threshold=0.5) -> pd.Series:
    """Share of each of the three classes that is flagged: recall for prediabetes and diabetes,
    false alarm rate for 'No diabetes'."""
    flagged = proba_fn(model)(X) >= threshold
    return pd.Series({CLASS_NAMES[c]: flagged[(y3 == c).to_numpy()].mean() for c in CLASS_NAMES}, name="share flagged")


def calibration(model, X, y, n_bins=10) -> pd.DataFrame:
    """Reliability diagram data: mean predicted risk vs observed share with (pre)diabetes per
    quantile bin. Only if they match may the score be read as a probability."""
    obs, pred = calibration_curve(y, proba_fn(model)(X), n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean predicted": pred, "observed": obs})


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
    p = pd.Series(proba_fn(model)(X), index=X.index, name="risk")
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


def logodds_fn(model):
    """Log-odds of the predicted risk; the unit of the logistic regression's weights."""
    f = proba_fn(model)

    def g(X):
        p = np.clip(f(X), 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))
    return g


class ShapExplainer:
    """SHAP with the model-agnostic PermutationExplainer and the same background sample for both
    models, so attributions are directly comparable. `output` is "probability" (default, used in
    every stakeholder figure) or "log-odds" (the logistic regression's own unit, used to compare
    with its exact contributions)."""

    def __init__(self, model, X_background: pd.DataFrame, n_background=200, seed=SEED, output="probability"):
        import shap
        self.model, self.seed = model, seed
        bg = X_background.sample(min(n_background, len(X_background)), random_state=seed)
        masker = shap.maskers.Independent(bg, max_samples=n_background)
        f = {"probability": proba_fn, "log-odds": logodds_fn}[output](model)
        self._exp = shap.PermutationExplainer(f, masker)

    def __call__(self, X: pd.DataFrame):
        """Returns a shap.Explanation (n_samples x n_features) for the positive class."""
        # PermutationExplainer reseeds numpy randomly when built without a seed, so seed before every
        # call - otherwise the neural network's SHAP values change between runs
        np.random.seed(self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = self._exp(X, silent=True)
        if ex.values.ndim == 3:  # (n, features, classes) -> class 1
            ex = ex[:, :, 1]
        ex.data = X.to_numpy()
        ex.feature_names = list(X.columns)
        return ex

    @staticmethod
    def global_importance(ex) -> pd.Series:
        return pd.Series(np.abs(ex.values).mean(0), index=ex.feature_names).sort_values(ascending=False)


class LimeExplainer:
    """LIME tabular explainer; local linear surrogate around one person.

    With discretize=True (LIME's default, used in Project 1) every non-binary feature is replaced by
    "is the person in the same quartile bin": the surrogate is linear in these bin indicators, not in
    the feature itself, so e.g. BMI 30 and BMI 45 look the same to it. With discretize=False the
    surrogate is linear in the standardised feature values."""

    def __init__(self, model, X_train: pd.DataFrame, seed=SEED, discretize=True):
        from lime.lime_tabular import LimeTabularExplainer
        self.model, self.discretize = model, discretize
        self._binary = [f for f in FEATURES if CODEBOOK[f].kind == "binary"]
        self._exp = LimeTabularExplainer(
            X_train.to_numpy(), feature_names=FEATURES, class_names=["No diabetes", "(Pre)diabetes"],
            categorical_features=[FEATURES.index(f) for f in self._binary],
            discretize_continuous=discretize, mode="classification", random_state=seed,
        )

    def explain(self, x: pd.Series, num_features=len(FEATURES), num_samples=5000):
        """Returns (contribution Series indexed by feature, raw LIME explanation).
        With bins, the surrogate's weight on "same bin as x" is already the contribution of x's value.
        Without bins, the weight is per standard deviation, so it is multiplied by x's standardised value
        (distance from the training mean) to give a contribution comparable with SHAP."""
        e = self._exp.explain_instance(x.to_numpy(), lambda Z: self.model.predict_proba(
            pd.DataFrame(Z, columns=FEATURES)), num_features=num_features, num_samples=num_samples)
        w = pd.Series({FEATURES[i]: v for i, v in e.as_map()[1]}).reindex(FEATURES).fillna(0.0)
        if not self.discretize:
            sc = self._exp.scaler
            z = (x.to_numpy() - sc.mean_) / sc.scale_
            cont = [f not in self._binary for f in FEATURES]
            w[cont] = w[cont] * z[cont]
        return w, e

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
    """Share of the k largest |attributions| that fall on `features` (e.g. the exact top features of the logistic regression)."""
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
# 6. TECHNICAL AUDIT: WHAT HAS THE NETWORK LEARNED (latent space, concepts)
# =============================================================================
def hidden_activations(model, X: pd.DataFrame, layer=0) -> np.ndarray:
    """Activations of hidden layer `layer` (0 = first) of the MLP pipeline: z_n = f_n(...f_1(x)).
    The same scaling and ReLU as in predict_proba, so these are what the output layer sees."""
    scaler, mlp = model[0], model[-1]
    Z = scaler.transform(X[model_columns(model)])
    for W, b in list(zip(mlp.coefs_, mlp.intercepts_))[:layer + 1]:
        Z = np.maximum(Z @ W + b, 0)
    return Z


def embed(Z: np.ndarray, method: str, seed=SEED, **kw) -> tuple[np.ndarray, dict]:
    """2-D embedding of Z with "PCA", "Random projection", "t-SNE" or "UMAP".
    Returns (n x 2 array, info). PCA reports the share of variance its two axes keep."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.random_projection import GaussianRandomProjection
    info = {}
    if method == "PCA":
        pca = PCA(n_components=2, random_state=seed).fit(Z)
        E = pca.transform(Z)
        info["explained variance"] = float(pca.explained_variance_ratio_.sum())
    elif method == "Random projection":
        E = GaussianRandomProjection(n_components=2, random_state=seed).fit_transform(Z)
    elif method == "t-SNE":
        E = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=seed, **kw).fit_transform(Z)
    elif method == "UMAP":
        import umap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E = umap.UMAP(n_components=2, random_state=seed, **kw).fit_transform(Z)
    else:
        raise ValueError(method)
    return E, info


def trustworthiness(Z: np.ndarray, E: np.ndarray, k=15) -> float:
    """How well the embedding keeps each point's k nearest neighbours (1 = perfectly; sklearn's
    trustworthiness). A functionally grounded check of how far a 2-D picture can be trusted."""
    from sklearn.manifold import trustworthiness as tw
    return float(tw(Z, E, n_neighbors=k))


def layer_representations(model, X: pd.DataFrame) -> dict:
    """Every representation of the network for the persons in X: the standardised inputs, each hidden
    layer and the output (log-odds)."""
    reps = {"Inputs": model[0].transform(X[model_columns(model)])}
    n_hidden = len(model[-1].coefs_) - 1
    for l in range(n_hidden):
        reps[f"Hidden {l + 1}"] = hidden_activations(model, X, layer=l)
    reps["Output"] = logodds_fn(model)(X)[:, None]
    return reps


# --- Concepts: probes, CAV and TCAV (Kim et al. 2018) -------------------------
def probe_auc(H: np.ndarray, concept: np.ndarray, seed=SEED) -> float:
    """How linearly readable a concept is from a representation: held-out AUC of a logistic regression
    that predicts the concept (0/1) from H (30 % held out). 0.5 = not encoded."""
    from sklearn.model_selection import train_test_split as tts
    Htr, Hte, ctr, cte = tts(H, concept, test_size=0.3, stratify=concept, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Htr, ctr)
    return float(roc_auc_score(cte, clf.predict_proba(Hte)[:, 1]))


def probe_report(H: np.ndarray, concept: np.ndarray, n_shuffle=10, n=1000, seed=SEED) -> dict:
    """Probe as in the week 11 exercise: equal numbers of concept and non-concept persons (at most `n` of
    each), a logistic regression trained on 70 % and its accuracy on the other 30 %. The same probe is
    trained `n_shuffle` times on shuffled concept labels; that random-label accuracy (about 0.5) is the
    baseline the real accuracy must beat."""
    from sklearn.model_selection import train_test_split as tts
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(concept == 1), np.flatnonzero(concept == 0)
    k = min(n, len(pos), len(neg))
    idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)]
    Hb, cb = H[idx], np.r_[np.ones(k), np.zeros(k)]

    def acc(labels, s):
        Htr, Hte, ctr, cte = tts(Hb, labels, test_size=0.3, stratify=labels, random_state=s)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Htr, ctr)
        return clf.score(Hte, cte)
    shuffled = [acc(rng.permutation(cb), int(rng.integers(1e9))) for _ in range(n_shuffle)]
    return {"probe accuracy": float(acc(cb, seed)), "random-label accuracy": float(np.mean(shuffled)),
            "random-label sd": float(np.std(shuffled)), "persons per group": int(k)}


CAV_METHODS = ["Mean difference", "Logistic regression", "Linear SVM", "PCA", "Contrastive PCA"]


def cav_direction(H_pos: np.ndarray, H_neg: np.ndarray, method="Logistic regression", seed=SEED) -> np.ndarray:
    """A concept direction in a layer's activation space, built in one of the five ways from the week 11
    exercise. Always a unit vector pointing from the non-concept mean towards the concept mean.
    - Mean difference: mean(concept) - mean(other)
    - Logistic regression / Linear SVM: the normal of the separating hyperplane (Kim et al. use the SVM)
    - PCA: first principal component of both groups together (maximises variance, not separation)
    - Contrastive PCA: first component of the concept group minus the first component of the other group"""
    from sklearn.decomposition import PCA
    from sklearn.svm import LinearSVC
    H = np.vstack([H_pos, H_neg])
    c = np.r_[np.ones(len(H_pos)), np.zeros(len(H_neg))]
    if method == "Mean difference":
        v = H_pos.mean(0) - H_neg.mean(0)
    elif method == "Logistic regression":
        v = LogisticRegression(max_iter=2000).fit(H, c).coef_[0]
    elif method == "Linear SVM":
        v = LinearSVC(max_iter=5000, random_state=seed).fit(H, c).coef_[0]
    elif method == "PCA":
        v = PCA(n_components=1, random_state=seed).fit(H).components_[0]
    elif method == "Contrastive PCA":
        v = (PCA(n_components=1, random_state=seed).fit(H_pos).components_[0]
             - PCA(n_components=1, random_state=seed).fit(H_neg).components_[0])
    else:
        raise ValueError(method)
    if v @ (H_pos.mean(0) - H_neg.mean(0)) < 0:
        v = -v
    return v / np.linalg.norm(v)


def cav_accuracy(v: np.ndarray, H_pos: np.ndarray, H_neg: np.ndarray) -> float:
    """How well the direction separates the two groups (week 11 `evaluate_cav`): project every person onto
    v and split at the median projection. 0.5 = no better than chance with equal group sizes."""
    s_pos, s_neg = H_pos @ v, H_neg @ v
    thr = np.median(np.r_[s_pos, s_neg])
    return float(((s_pos > thr).sum() + (s_neg <= thr).sum()) / (len(s_pos) + len(s_neg)))


def concept_activation_vector(H_concept: np.ndarray, H_other: np.ndarray, method="Logistic regression",
                              seed=SEED) -> tuple[np.ndarray, float]:
    """CAV built on 70 % of the persons with `cav_direction`; returns (CAV, accuracy on the other 30 %)."""
    from sklearn.model_selection import train_test_split as tts
    H = np.vstack([H_concept, H_other])
    c = np.r_[np.ones(len(H_concept)), np.zeros(len(H_other))]
    Htr, Hte, ctr, cte = tts(H, c, test_size=0.3, stratify=c, random_state=seed)
    v = cav_direction(Htr[ctr == 1], Htr[ctr == 0], method, seed)
    return v, cav_accuracy(v, Hte[cte == 1], Hte[cte == 0])


def logit_gradient(model, X: pd.DataFrame, layer=0) -> np.ndarray:
    """Gradient of the output log-odds with respect to the activations of hidden layer `layer`, one row
    per person (backpropagation through the later ReLU layers). For the LAST hidden layer this is the
    output weight vector for everyone, so TCAV there can only give 0 or 1."""
    mlp = model[-1]
    Ws, bs = mlp.coefs_, mlp.intercepts_
    Z = model[0].transform(X[model_columns(model)])
    pre = []
    for W, b in zip(Ws[:-1], bs[:-1]):
        z = Z @ W + b
        pre.append(z)
        Z = np.maximum(z, 0)
    g = np.tile(Ws[-1][:, 0], (len(X), 1))            # d logit / d a_last
    for k in range(len(Ws) - 2, layer, -1):           # back through hidden layers above `layer`
        g = (g * (pre[k] > 0)) @ Ws[k].T
    return g


def tcav(model, X_concept: pd.DataFrame, X_pool: pd.DataFrame, X_target: pd.DataFrame, layer=0,
         n_runs=20, n=300, method="Logistic regression", seed=SEED) -> dict:
    """TCAV score of a concept in hidden layer `layer`: the share of target persons (e.g. persons with
    (pre)diabetes) whose predicted log-odds increase when their activations move towards the concept.
    Repeated with `n_runs` random counterexample sets; the same is done for random 'concepts' (random
    persons vs random persons) to get a null distribution, and a two-sided t-test compares the two
    (Kim et al. 2018). `method` is one of CAV_METHODS. Also returns the CAV's held-out accuracy and the
    CAVs themselves (`cavs`), so directions from different methods can be compared."""
    from scipy.stats import ttest_ind
    rng = np.random.default_rng(seed)
    g = logit_gradient(model, X_target, layer)
    pool_H = hidden_activations(model, X_pool, layer)
    conc_H = hidden_activations(model, X_concept, layer)
    scores, null, accs, cavs = [], [], [], []
    for _ in range(n_runs):
        ci = rng.choice(len(conc_H), min(n, len(conc_H)), replace=False)
        r1, r2 = (rng.choice(len(pool_H), n, replace=False) for _ in range(2))
        v, acc = concept_activation_vector(conc_H[ci], pool_H[r1], method, seed=int(rng.integers(1e9)))
        vr, _ = concept_activation_vector(pool_H[r2], pool_H[r1], method, seed=int(rng.integers(1e9)))
        scores.append(float((g @ v > 0).mean()))
        null.append(float((g @ vr > 0).mean()))
        accs.append(acc)
        cavs.append(v)
    p = ttest_ind(scores, null).pvalue
    return {"TCAV score": np.mean(scores), "TCAV sd": np.std(scores), "random score": np.mean(null), "random sd": np.std(null),
            "p-value": float(p), "CAV accuracy": np.mean(accs), "scores": scores, "null": null, "cavs": np.array(cavs)}


# --- Removing a concept inside the network (ablation) -------------------------
def forward_from_layer(model, H: np.ndarray, layer=0) -> np.ndarray:
    """Predicted probability when hidden layer `layer` is replaced by the activations H: the rest of the
    network (later layers, ReLU, output) is run unchanged. With H = hidden_activations(model, X, layer)
    this equals predict_proba."""
    mlp = model[-1]
    Ws, bs = mlp.coefs_, mlp.intercepts_
    Z = H
    for k in range(layer + 1, len(Ws)):
        Z = Z @ Ws[k] + bs[k]
        if k < len(Ws) - 1:
            Z = np.maximum(Z, 0)
    return 1 / (1 + np.exp(-Z[:, 0]))


def concept_ablation(model, X_members: pd.DataFrame, X_concept: pd.DataFrame, X_other: pd.DataFrame,
                     X_pool: pd.DataFrame, layer=0, method="Logistic regression", n_runs=10, n=300,
                     seed=SEED) -> dict:
    """What the network loses when a concept is removed from a hidden layer. For each person in X_members
    (persons who have the concept) the coordinate along the CAV is moved to the average coordinate of
    persons without the concept; everything orthogonal to the CAV is kept, and the rest of the network is
    run with `forward_from_layer`. The CAV is built `n_runs` times (n persons per group, as in `tcav`) and
    the changes are averaged.
    The same is done along CAVs of random 'concepts' (random vs random persons from X_pool, moved to the
    mean of one random group): the change that removing an arbitrary direction causes.
    Returns per-person changes in predicted probability (`change`, `random change`) and their means."""
    rng = np.random.default_rng(seed)
    H = hidden_activations(model, X_members, layer)
    p0 = forward_from_layer(model, H, layer)
    Hc, Ho, Hp = (hidden_activations(model, X, layer) for X in (X_concept, X_other, X_pool))

    def remove(v, target_mean):
        return forward_from_layer(model, H + (target_mean - H @ v)[:, None] * v, layer) - p0
    d, dn = [], []
    for _ in range(n_runs):
        ci = rng.choice(len(Hc), min(n, len(Hc)), replace=False)
        oi = rng.choice(len(Ho), min(n, len(Ho)), replace=False)
        v = cav_direction(Hc[ci], Ho[oi], method, int(rng.integers(1e9)))
        d.append(remove(v, (Ho @ v).mean()))
        r1, r2 = rng.choice(len(Hp), n, replace=False), rng.choice(len(Hp), n, replace=False)
        vr = cav_direction(Hp[r1], Hp[r2], method, int(rng.integers(1e9)))
        dn.append(remove(vr, (Hp[r2] @ vr).mean()))
    d, dn = np.mean(d, axis=0), np.mean(dn, axis=0)
    return {"persons": len(H), "risk before": p0.mean(), "change": d, "random change": dn,
            "mean change": d.mean(), "share whose risk falls": (d < 0).mean(),
            "random mean absolute change": np.abs(dn).mean()}


# --- Concepts nobody defined: NMF (week 11, exercise 3) ----------------------
def nmf_concepts(H: np.ndarray, k=8, seed=SEED, init="nndsvda") -> tuple[np.ndarray, np.ndarray, float]:
    """Non-negative matrix factorisation of a ReLU layer, H (persons x units) ~ U W.
    W (k x units) holds k concept directions, U (persons x k) how strongly each person expresses each.
    Returns (U, W with unit rows, relative reconstruction error ||H - UW|| / ||H||)."""
    from sklearn.decomposition import NMF
    m = NMF(n_components=k, init=init, random_state=seed, max_iter=3000).fit(np.clip(H, 0, None))
    U, W = m.transform(H), m.components_
    err = float(np.linalg.norm(H - U @ W) / np.linalg.norm(H))
    norms = np.linalg.norm(W, axis=1)
    return U * norms, W / norms[:, None], err


def nmf_stability(H: np.ndarray, W_ref: np.ndarray, n_seeds=5) -> np.ndarray:
    """How reproducible each concept in W_ref is: NMF is refitted from `n_seeds` random starts, the new
    concepts are matched one-to-one to W_ref (Hungarian matching on cosine similarity), and the mean
    cosine of each reference concept with its match is returned (1 = found again every time)."""
    from scipy.optimize import linear_sum_assignment
    cos = []
    for s in range(n_seeds):
        _, W, _ = nmf_concepts(H, len(W_ref), seed=SEED + 1 + s, init="random")
        C = W_ref @ W.T
        r, c = linear_sum_assignment(-C)
        cos.append(C[r, c][np.argsort(r)])
    return np.mean(cos, axis=0)


def concept_profile(X: pd.DataFrame, member: np.ndarray, n_features=3) -> str:
    """Who is in a group of persons, in words: the features that differ most from everyone in X
    (in standard deviations), as 'group value (all value)'; shares for yes/no answers, the answer
    closest to the mean for coded answers (age group, general health, ...), means otherwise."""
    z = (X[member].mean() - X.mean()) / X.std()
    out = []
    for f in z.abs().sort_values(ascending=False).index[:n_features]:
        g, a = X.loc[member, f].mean(), X[f].mean()
        name = "Men" if f == "Sex" else label(f)
        if set(np.unique(X[f])) <= {0, 1}:
            out.append(f"{name} {100 * g:.0f} % (all {100 * a:.0f} %)")
        elif CODEBOOK[f].values:
            out.append(f"{name} {describe_value(f, round(g))} (all {describe_value(f, round(a))})")
        else:
            out.append(f"{name} {g:.1f} (all {a:.1f})")
    return "; ".join(out)


def leading_feature(X: pd.DataFrame, member: np.ndarray) -> tuple[str, int]:
    """The feature that sets a group apart most, and the direction (+1 higher, -1 lower than everyone)."""
    z = (X[member].mean() - X.mean()) / X.std()
    f = z.abs().idxmax()
    return f, int(np.sign(z[f]))


# =============================================================================
# 7. PLOT STYLE + STAKEHOLDER FIGURES
# =============================================================================
# One colour system for every figure (week 8: consistent colours, colour-blind safe):
#   red / green            raises / lowers the predicted risk (bluish green stays apart from red with
#                          colour blindness; bars on opposite sides of 0 as a second cue)
#   green -> yellow -> red the patient's score below / around / above the average person
#   dark / light purple    logistic regression / neural network
#   black / blue / grey    exact contributions (truth) / SHAP / LIME
#   light grey             neutral (correct predictions, no diabetes)
#   orange / dark blue     prediabetes / diabetes (latent-space plots)
#   cividis                feature value in the beeswarm (no red or green)
COLORS = {
    "surface": "white", "text": "black", "text2": "dimgray", "grid": "lightgray",
    "increase": "#d62728",    # pushes risk UP
    "decrease": "#009e73",    # pushes risk DOWN
    "neutral": "#c8c8c8",
}
MODEL_COLORS = {"Logistic regression": "#542788", "Neural network": "#b2abd2"}
METHOD_COLORS = {"Exact (truth)": "black", "SHAP": "#1f77b4", "LIME": "#a6a6a6"}
CLASS_COLORS = {"No diabetes": COLORS["neutral"], "Prediabetes": "#e69f00", "Diabetes": "#23366f"}
FEATURE_CMAP = "cividis"
SCORE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "score", [COLORS["decrease"], "#fee08b", "#f46d43", COLORS["increase"]])
SCORE_SPAN = 25          # score points from the average person to full green / full red


def set_style():
    plt.style.use("default")
    mpl.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})


def savefig(fig, stakeholder: str, name: str):
    """figures/<stakeholder>/<name>.png - one folder per audience."""
    d = FIG_DIR / stakeholder
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", bbox_inches="tight")


def same_xlim(*axes):
    """Shared x-axis for panels that are compared with each other."""
    lo = min(a.get_xlim()[0] for a in axes)
    hi = max(a.get_xlim()[1] for a in axes)
    for a in axes:
        a.set_xlim(lo, hi)


def micro_legend(ax, colors: dict, title=""):
    """One-line legend directly above the plot (under the title), in the same order as the bars."""
    handles = [mpl.patches.Patch(color=c, label=name) for name, c in colors.items()]
    ax.legend(handles=handles, ncol=len(handles), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
              fontsize=10, handlelength=1.2, borderaxespad=0.3, columnspacing=1.5)
    ax.set_title(title, pad=28 if title else 20)


def plot_contributions(ax, contrib: pd.Series, title="", xlabel="Change in predicted risk (percentage points)",
                       plain_labels=True, max_features=10, min_abs=0.0, values: pd.Series | None = None):
    """Horizontal diverging bars: red = raises risk, green = lowers risk.
    `min_abs` hides features with (near) zero contribution, e.g. features a model hardly uses.
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
    """One horizontal bar per column (model or method) for every row, first column on top, with a
    micro-legend above the plot. Colours from MODEL_COLORS / METHOD_COLORS unless `colors` is given."""
    colors = {**MODEL_COLORS, **METHOD_COLORS, **(colors or {})}
    n = values.shape[1]
    h = 0.8 / n
    y = np.arange(len(values))
    for i, name in enumerate(values.columns):
        ax.barh(y + ((n - 1) / 2 - i) * h, values[name], height=h, color=colors.get(name))
    ax.set_yticks(y, values.index)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=xlabel)
    micro_legend(ax, {name: colors.get(name) for name in values.columns}, title)
    return ax


def plot_points(ax, points: pd.Series, title="", max_features=12, unreliable=()):
    """The points table as bars: red = adds points (higher risk), green = subtracts points.
    Features in `unreliable` (unsafe advice, or a pattern that is not a cause) are grey and labelled,
    so a minus for e.g. heavy drinking does not read as 'good for you'."""
    pts = points.reindex(points.abs().sort_values(ascending=False).index)[:max_features][::-1]
    grey = [f in unreliable for f in pts.index]
    ax.barh([point_label(f) + ("  (not reliable)" if g else "") for f, g in zip(pts.index, grey)], pts.values,
            color=[COLORS["neutral"] if g else COLORS["increase"] if v > 0 else COLORS["decrease"]
                   for v, g in zip(pts.values, grey)], height=0.6)
    for i, (v, g) in enumerate(zip(pts.values, grey)):
        ax.text(v + (0.3 if v > 0 else -0.3), i, f"{v:+d}", va="center", ha="left" if v > 0 else "right",
                color=COLORS["text2"] if g else COLORS["text"])
    ax.axvline(0, color=COLORS["text2"], lw=1)
    lim = pts.abs().max() + 2
    ax.set(xlim=(-lim, lim), xlabel="Points", title=title)
    ax.grid(axis="y", visible=False)
    return ax


def plot_embedding(ax, E: np.ndarray, classes: np.ndarray | None = None, values: np.ndarray | None = None,
                   title="", cmap=FEATURE_CMAP, s=4):
    """Scatter of a 2-D embedding, coloured by the three classes (no diabetes drawn first, so the small
    classes stay visible) or by a continuous value. The axes of t-SNE and UMAP have no unit, so no ticks."""
    if classes is not None:
        for c, name in CLASS_NAMES.items():
            m = classes == c
            ax.scatter(E[m, 0], E[m, 1], s=s, color=CLASS_COLORS[name], lw=0, label=name, rasterized=True)
    else:
        sc = ax.scatter(E[:, 0], E[:, 1], s=s, c=values, cmap=cmap, lw=0, rasterized=True)
        ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    ax.set(xticks=[], yticks=[], title=title)
    ax.grid(False)
    return ax


# Not actionable and socio-economic: shown to the patient as one summed bar (SHAP values are additive).
BACKGROUND = ["Sex", "Income", "Education"]
BACKGROUND_LABEL = "Background (sex, income, education)"


def plot_patient_factors(ax_habits, ax_other, contrib: pd.Series, x: pd.Series, hide=()):
    """The patient's own contributions in two panels with a shared axis: habits they can change, and
    the rest (health and background, with sex/income/education summed into one bar)."""
    habits = contrib[[f for f in ACTIONABLE if f not in hide]]
    other = contrib[[f for f in FEATURES if CODEBOOK[f].actionable != "yes" and f not in BACKGROUND + list(hide)]].copy()
    other[BACKGROUND_LABEL] = contrib[BACKGROUND].sum()
    plot_contributions(ax_habits, habits, title="Things you can change yourself",
                       xlabel="Effect on your score (points)", max_features=5, min_abs=1e-3, values=x)
    if (habits.abs() <= 1e-3).all():
        ax_habits.text(0.5, 0.5, "None of your habits change\nthis model's score", transform=ax_habits.transAxes,
                       ha="center", va="center", color=COLORS["text2"])
    plot_contributions(ax_other, other, title="Your health and background",
                       xlabel="Effect on your score (points)", max_features=5, min_abs=1e-3, values=x)
    same_xlim(ax_habits, ax_other)


def plot_patient_changes(ax, cf: pd.DataFrame, average: float):
    """The patient's score after each realistic change (counterfactual path), against the average person.
    Bars take the colour of their score (green below, yellow around, red above the average person) and
    the same scale is faded into the background."""
    avg = round(100 * average)
    steps = cf.copy(); steps["risk"] *= 100
    norm = mpl.colors.TwoSlopeNorm(vcenter=avg, vmin=avg - SCORE_SPAN, vmax=avg + SCORE_SPAN)
    shade = lambda s: SCORE_CMAP(np.clip(norm(s), 0, 1))
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.imshow(shade(np.linspace(0, 100, 256)).reshape(-1, 1, 4), extent=(-0.5, len(steps) - 0.5, 0, 100),
              origin="lower", aspect="auto", alpha=0.18, zorder=0)
    ax.bar(range(len(steps)), steps["risk"], color=[shade(r) for r in steps["risk"]], width=0.6,
           edgecolor=COLORS["text2"], lw=0.8, zorder=2)
    for i, r in enumerate(steps["risk"]):
        ax.text(i, r + 1.5, f"{r:.0f}", ha="center", fontweight="bold", zorder=3)
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


def patient_figure(contrib: pd.Series, x: pd.Series, cf: pd.DataFrame, risk: float, average: float,
                   prevalence: float, hide=(), subtitle=""):
    """Three panels for the patient: what raises/lowers *their* score (things they can change
    vs everything else) and the effect of realistic changes.

    Design choices (good explanations are selected, contrastive and truthful):
    - a score out of 100 compared with the average person (the SHAP base value);
    - only habits count as "things you can change". High blood pressure / cholesterol are
      "ever told you have ..." answers - they belong with the doctor, not in a to-do list;
    - sex, income and education are summed into one background bar: not actionable, and
      "your income raises your risk" can stigmatise without helping;
    - features in `hide` (flagged unsafe by advice_safety) are never shown."""
    score, avg = round(100 * risk), round(100 * average)
    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.75)
    plot_patient_factors(fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), contrib, x, hide)
    plot_patient_changes(fig.add_subplot(gs[2]), cf, average)

    fig.suptitle(f"Your diabetes risk score: {score} out of 100", fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.06)
    compare = "higher than" if score > avg else "lower than" if score < avg else "the same as"
    fig.text(0.01, 0.97, f"This is {compare} the average person in our data (score {avg}). {subtitle}",
             ha="left", color=COLORS["text2"])
    note = ("How to read this: red bars raise your score, green bars lower it. On the right, green is below the average person, "
            "yellow is around average and red is above. Each bar compares your answer with the average person, "
            "so an answer close to average gives a small bar, even if changing it lowers your score (right). "
            f"The score shows how much your answers resemble those of people in our data who have diabetes or prediabetes "
            f"(about {round(100 * prevalence)} in 100 of them). It is not a diagnosis. The bars show what the model reacts to, "
            "not proven causes. Talk to your doctor about blood pressure, cholesterol and your general health.")
    if hide:
        note += " Some habits are left out because the model is not reliable about them."
    fig.text(0.01, -0.03, textwrap.fill(note, 150), ha="left", va="top", color=COLORS["text2"], fontsize=9)
    return fig


RISK_CMAP = mpl.colors.LinearSegmentedColormap.from_list("risk", [COLORS["decrease"], "white", COLORS["increase"]])


def plot_heatmap(ax, values: pd.DataFrame, title="", vmin=0.0, vmax=1.0, fmt="{:.2f}", cmap="Greys"):
    """Annotated heatmap (probe AUCs, CAV similarities, ablation effects). Grey scale, so it does not borrow a colour that already
    has a meaning elsewhere (red/green risk, purple models, blue SHAP). With cmap=RISK_CMAP and vmin = -vmax it
    shows changes in risk: red = raises the risk, green = lowers it, as in every other figure."""
    im = ax.imshow(values.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(values.shape[1]), values.columns)
    ax.set_yticks(range(values.shape[0]), values.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.iat[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=("white" if abs(v) / vmax > 0.6 else "black") if cmap is RISK_CMAP
                    else "white" if (v - vmin) / (vmax - vmin) > 0.55 else "black", fontsize=10)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.set_title(title)
    return im


def plot_tcav(ax, results: pd.DataFrame, runs: dict, alpha=0.05, title=""):
    """TCAV score per concept (mean ± sd over the runs), with the score of every random direction as a
    small grey dot (`runs[concept]["null"]`, jittered). Red = the concept significantly raises the
    predicted risk, green = lowers it, grey = not different from the random directions. Random directions
    mostly score near 0 or 100, because the persons' gradients point roughly the same way."""
    r = results.iloc[::-1]
    y = np.arange(len(r))
    rng = np.random.default_rng(SEED)
    for i, c in zip(y, r.index):
        null = np.asarray(runs[c]["null"]) * 100
        ax.scatter(null, i + rng.uniform(-0.25, 0.25, len(null)), s=12, color=COLORS["neutral"], zorder=1, lw=0)
    sig = r["p-value"] < alpha
    colors = [COLORS["neutral"] if not s else COLORS["increase"] if v > rs else COLORS["decrease"]
              for s, v, rs in zip(sig, r["TCAV score"], r["random score"])]
    ax.scatter(r["TCAV score"] * 100, y, color=colors, s=90, zorder=3, edgecolor=COLORS["text2"])
    ax.errorbar(r["TCAV score"] * 100, y, xerr=r["TCAV sd"] * 100, fmt="none", ecolor=COLORS["text2"], zorder=2)
    ax.axvline(50, color=COLORS["text2"], lw=1, ls="--")
    ax.set_yticks(y, r.index)
    ax.set(xlim=(-2, 102), xlabel="TCAV score (% of persons with (pre)diabetes whose risk rises towards the concept)",
           title=title)
    ax.grid(axis="y", visible=False)
    return ax
