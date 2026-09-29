# XAI Projekt – Auditing and Explaining Diabetes Risk Prediction Models

Assignment 1, Explainable AI and UX (DTU). We train several diabetes risk models on the CDC BRFSS 2015 data
(`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`), audit them with XAI methods and build
explanations for three stakeholders: head of data science, director of the patient organisation, and patient.

## Contents
- `projekt.ipynb` – main notebook: data, models, global/local explanations, validation of explanations,
  bias and error analysis, counterfactuals, stakeholder figures
- `xai_framework.py` – reusable code (codebook, models, SHAP/LIME/permutation importance, validation checks, plot style)
- `figures/<stakeholder>/` – exported figures per audience

## Models
Logistic regression, random forest, gradient boosting (HistGradientBoosting) and a neural network (MLP).

## Setup
```bash
pip install pandas numpy scikit-learn shap lime matplotlib scipy jupyter
```
Then open `projekt.ipynb` and run all cells (about 3 minutes).
