# XAI Projekt – Auditing and Explaining Diabetes Risk Prediction Models

Assignment 1, Explainable AI and UX (DTU). We train two diabetes risk models on the CDC BRFSS 2015 data
(`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`), audit them with XAI methods and build
explanations for three stakeholders: head of data science, director of the patient organisation, and patient.

## Contents
- `projekt.ipynb` – main notebook: data, the two models and their settings, global/local explanations, validation of explanations
  (incl. a summary of the validation scheme), bias and error analysis,
  counterfactuals, stakeholder figures, draft recommendation to the client
- `xai_framework.py` – reusable code (codebook, models, SHAP/LIME, validation checks, plot style and figure helpers)
- `figures/<stakeholder>/` – exported figures per audience

## Models
- **Logistic regression** – transparent; its weights are the explanation, and its exact per-feature contributions
  serve as ground truth for checking the explanation methods. (A depth-5 decision tree was tried first and replaced:
  it could hardly give lifestyle advice.)
- **Neural network** (MLP) – black box, explained post hoc.

## Explanation methods
- **SHAP** – main method: local and global explanations for both models, in percentage points of risk.
- **LIME** – second local method, used to validate SHAP (agreement, faithfulness, stability).
- Both are also averaged over 100 persons for a global comparison with the exact contributions.

## Setup
```bash
pip install pandas numpy scikit-learn shap lime matplotlib scipy jupyter
```
Then open `projekt.ipynb` and run all cells.
