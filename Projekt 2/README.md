# Project 2: Auditing and explaining diabetes risk prediction models

02820 Explainable AI and user experience, group 4. Presentation 1.12, report due 3.12 at 18:00, oral exam 11.12.

## What is in this folder

| File | What it is |
|---|---|
| `projekt2.ipynb` | The whole analysis, run from top to bottom (about 12 minutes). Every section ends with a short text on what the results mean. |
| `xai_framework.py` | All reusable code (data, models, SHAP/LIME, validation, audit methods, plot style). The notebook calls it as `xf.`. |
| `EXAM_OVERVIEW.md` | **Read this first.** Every choice in Project 1 and Project 2 with the reason for it, everything that looked odd and why, and the challenges per project. Written so all of us can answer "why did you do X?" at the exam. |
| `figures/` | All figures, one folder per stakeholder (`data_scientist`, `director`, `patient`). |
| `diabetes_012_health_indicators_BRFSS2015.csv` | The data (253,680 persons, three classes, real prevalence). |

## How to run
Python 3.11 with numpy, pandas, scikit-learn 1.7, matplotlib, scipy, shap 0.51, lime and umap-learn. Open
`projekt2.ipynb` from this folder and run all cells.

## What has been done (notebook sections)

1. **Data and split (2-3).** Diabetes_012 with 15.8 % positives. We predict prediabetes or diabetes against neither
   (binary). This is our own choice, not required by the description: prediabetes has no region of its own in the
   network (12.1), screening leads to the same next step for both (a blood test), and one calibrated score is easier
   to explain. The three classes are kept for stratifying the split and for recall per class.
2. **Models (4).** Logistic regression (C = 1) and a neural network (128 + 64 units, L2 penalty 0.01). Imbalance is
   handled by choosing the threshold on the validation set, not by resampling or class weights, so the score stays
   a real risk (calibration in 4.3). Two threshold rules are compared: Youden's J (used now) and recall 80 %.
3. **Explanations (5-6).** SHAP for both models, LIME as a comparison, global and for one example person.
4. **Validating the explanations (7).** SHAP against the logistic regression's exact contributions, units and bins
   (the Project 1 feedback), deletion test, stability, and whether the advice is safe. Heavy drinking still "protects"
   in both models; a version without it is tested.
5. **Errors and bias (8).** Missed cases per age, sex, income and education; models retrained without sex, income
   and education.
6. **What the patient can change (9)** and the **stakeholder figures (10)** for the data scientist, the director and
   the patient.
7. **Technical audit of the network (12).**
   - 12.1 Latent space: inputs and hidden layers in 2-D (PCA, random projection, t-SNE, UMAP).
   - 12.2 Concepts: probes, CAVs built five ways, TCAV, and NMF to find concepts we did not define.
   - 12.3 Sensitive attributes: the network trained without sex, income and education still encodes them.
   - 12.4 Ablation: removing a concept inside the network to measure how much of the score it carries.
   - 12.5 A deeper network (4 hidden layers): does depth help, and where do concepts form?
   - 12.6 Summary table of the whole audit.

Sections 12.4 and 12.5 are our "challenge" (the teacher asked us to challenge ourselves somewhere). They go beyond
the course methods, so the notebook explains every choice made there.

## Main results in short
- AUC 0.815 (logistic regression) and 0.822 (network); both well calibrated. About 78 of 100 persons with
  (pre)diabetes are caught at about 29 false alarms per 100; young persons are mostly missed.
- The network's score is driven by the metabolic profile (high blood pressure, high cholesterol, BMI 30+), poor
  general health and cardiovascular history. Removing the metabolic profile lowers its members' risk by 18 points.
- Social circumstances (low income, barriers to care) raise the risk a little (1.5-3.4 points). Removing income,
  sex and education from the inputs does not remove them from the network.
- Heavy drinking lowers the predicted risk in both models (reverse causation in survey data) and a healthy lifestyle
  is barely rewarded, so the patient view must never advise on these.
- More layers give the same AUC (0.820-0.822) and make the network harder to audit.

## Still open (to decide together)
- Threshold rule: Youden's J or recall 80 %.
- Which model to recommend for deployment (in any case the version without heavy drinking).
- Validation summary (7.7) and recommendation text (11).
- User Story Maps for all three stakeholders, Before/After designs, think-aloud study, report and presentations.
