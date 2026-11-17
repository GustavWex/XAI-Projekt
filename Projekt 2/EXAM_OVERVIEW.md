# Exam overview: decisions, oddities and challenges in Project 1 and Project 2

Numbers are taken from the Project 1 report (handed in 24.10) and from `projekt2.ipynb` as it runs on 17.11.
Project 2 numbers use the Youden's J threshold unless stated otherwise (the threshold rule is still open).

---

## Part A. Project 1 (50/50 data, 70,692 persons)

### A.1 Setup in one paragraph
Predict (pre)diabetes from 21 survey answers (BRFSS 2015, balanced 50/50 version). Logistic regression (C = 1) and a
neural network (one hidden layer of 32 ReLU units, alpha 1e-4, Adam, early stopping). Test AUC 0.823 vs 0.828, McNemar
p ≈ 0.02. SHAP (PermutationExplainer, probability units) as the main method, LIME (tabular, quartile bins) as a check.
Explanations designed for the head of data science, the director of the patient organisation and the patient.

### A.2 Deliberate choices and why

| Choice | Why |
|---|---|
| Logistic regression as one of the two models | Transparent: one weight per feature, so its exact contribution per person (weight × (value − background mean)) is a ground truth for checking SHAP and LIME |
| Neural network as the other | A black box that can learn interactions, which is the case XAI is meant for |
| Small network (32 units) | Nine settings gave validation AUC 0.831-0.833; we took the smallest |
| C = 1 for the logistic regression | All regularisation strengths gave 0.828 on validation |
| 70/10/20 split, stratified, standardised on train only | Settings chosen on validation, test untouched; no information from validation/test leaks into the scaling |
| SHAP as main method | Contributions add up exactly to score minus average (additivity), works for both models, from global to local views |
| LIME only as a check | Its answer depends on sampling and binning; local fit was poor (mean R² ≈ 0.49) |
| Functionally grounded validation (Doshi-Velez & Kim) | No users available; we test the explanations against the models: ground truth (LR), deletion test, stability, plausibility |
| Advice-safety check | Give everyone the healthy version of each habit and measure the risk change; a healthy change must not raise the risk |
| Retrain without heavy drinking | The fix for the heavy-drinking problem (see A.3): AUC barely changed, missed heavy drinkers halved |
| Keep sex, income, education | Removing them barely changed AUC and did not remove the group differences, so they are kept, but summed into one bar for the patient |
| Recommend the logistic regression | Almost as accurate, and its explanation can be checked exactly |
| Patient view: score vs average person, habits separated from health and background, own answer in every label, realistic changes (BMI 29 → 24 → 22, exercise), heavy drinking never advised | Reduce misreading: a bar is the effect of the person's own answer, only changeable things are advice |
| Director view: per-100 counts and a points table (1 point = 0.1 log-odds) | No metrics or weights for a non-technical reader |
| Conceptual model: the score describes survey patterns, is not a diagnosis, not a future risk, bars are not causes | Norman: users form their own model from our figures, so we state the one we want them to form |

### A.3 What was odd

| Oddity | What we saw | Likely reason | What we did |
|---|---|---|---|
| **Heavy drinking protects** | Stopping heavy drinking raised predicted risk ~12 pp in both models; 53 % of heavy drinkers with (pre)diabetes missed vs ~20 % of others; points table −7 | Reverse causation in survey data: people already diagnosed are told to drink less, so current heavy drinkers are mostly people not (yet) diagnosed | Never advise on it; retrained without it (AUC 0.823 → 0.822 and 0.828 → 0.824, missed heavy drinkers 53 % → ~25 %) |
| **Cholesterol check raises the risk** | Points table +13, the largest; LIME gave it a very large weight for the example person | Having had a check means contact with a doctor, which is also what it takes to be told you have diabetes; 96 % have had one, so "no" is rare | Explained in the report; Project 2 explains the LIME part properly (B.4) |
| **LIME flips BMI** | LIME said BMI 29 lowers the risk, while risk rises with BMI | LIME compares the person's quartile bin (25-29) with all other bins; the bins above 29 have much higher risk. Teachers corrected us: this is our surrogate setup (`discretize_continuous=True`), not "how LIME works" | Project 2 runs LIME with and without bins |
| **SHAP and LIME disagree** | Top-5 overlap with the truth: SHAP 0.97, LIME 0.48 | LIME's local linear model on binned, independently sampled features | Trust SHAP, LIME only as a direction check |

### A.4 Challenges

1. **50/50 data.** Scores are not real risks: the "average person" scored 48. Peers asked whether the dataset is
   representative; we could not say the scores were calibrated.
2. **Features pointing the wrong way** (heavy drinking, cholesterol check), so the models could give harmful advice.
3. **No ground truth for the network.** Solved indirectly: validate on the LR (where the truth is known), then use
   the deletion test, which works for any model (SHAP top feature changed the prediction 14-15 pp, LIME 2-6, random 3).
4. **Which explanation to trust** when SHAP and LIME disagree.
5. **Sensitive features.** Removing sex, income, education did not remove the group differences: other features act
   as proxies.
6. **Young people mostly missed** (few cases in those groups).
7. **Non-technical users** reading bars as causes or the score as a diagnosis.
8. Found through feedback: SHAP (probability) and the exact contributions (log-odds) were in different units; % shown
   both as 0-1 and 0-100; colours not consistent; model chosen before the users' goals were analysed.

---

## Part B. Project 2 (real prevalence, 253,680 persons)

### B.1 What carried over from Project 1 unchanged
- The two model types and their roles (LR = transparent reference, NN = black box), LR with C = 1.
- 70/10/20 split, standardisation on train, settings chosen on validation AUC.
- SHAP as the main method, LIME as a check, and the whole functionally grounded validation suite (ground truth from
  the LR, deletion test, stability, advice safety).
- The heavy-drinking fix (retrain without it) and never giving advice on unsafe features.
- Keeping sex, income and education, and testing their removal.
- All stakeholder design rules (score vs average, separated habits, summed background bar, own answer in labels,
  realistic counterfactuals with BMI floor 22, per-100 counts, points table).
- The redesigned colour system: red/green = raises/lowers risk, dark/light purple = LR/NN, black/blue/grey =
  exact/SHAP/LIME, grey = neutral, cividis for feature values, greyscale heatmaps.
- Fixed seeds; numpy reseeded before every SHAP call (otherwise the NN figures change between runs).

### B.2 New decisions and why

**Data and target**

| Decision | Why |
|---|---|
| Binary target: prediabetes + diabetes vs no diabetes (15.8 % positive) | **Our choice, not required** (the description is open, and the teacher said to pick the most defensible option). Reasons: prediabetes is only 1.8 % and has no region of its own in the latent space; in screening both lead to the same next step (a blood test); one calibrated score keeps the explanations simple for the patient |
| Keep the three-class label for stratifying the split and for recall per class | So prediabetes is spread evenly, and so we can show that it is caught less often (65-68 per 100) |
| Train on the real distribution, no resampling, no class weights | Keeps the score calibrated (a score of 23 means about 23 in 100). Class weights: same recall/specificity but mean predicted risk 39.4 % vs 15.8 % observed. Over/undersampling tested: AUC same or worse (oversampled NN 0.780), calibration breaks. Positives are plentiful (~28,000 in training) |
| Kept 25,772 exact duplicate rows | Plausible with mostly yes/no answers; they are different respondents |

**Models and threshold**

| Decision | Why |
|---|---|
| Threshold chosen on validation instead of 0.5 | At 0.5 the models catch only ~18 of 100 cases |
| Two threshold rules compared: Youden's J (LR 0.157, NN 0.154) and recall 80 % (LR 0.136, NN 0.145) | **Open.** Youden: 76/78 caught per 100, 28/29 false alarms. Recall 80 %: 80 caught, 32/31 false alarms. Recommendation: recall 80 % (screening, easy to explain, both models at the same recall, in the course material). Per 1000 screened (NN): +3 caught for +17 false alarms |
| Network 128 + 64 units, alpha 0.01 | AUC 0.820-0.822 for all nine settings, so AUC does not decide. Two layers let us follow the representation through the network (latent space), and TCAV needs a layer whose gradient differs between persons: in the last hidden layer the gradient is the output weight vector for everyone, so TCAV can only give 0 or 1 |
| New metrics: average precision, balanced accuracy, recall per class, calibration, McNemar | Accuracy is meaningless (84 % by always saying "no") |
| Calibration check (reliability diagram) | Only a calibrated score may be shown as "risk out of 100"; both models lie close to the diagonal |

**Explanations (answers to the feedback)**

| Decision | Why |
|---|---|
| LR weights shown as log-odds ratios and odds ratios | Feedback: "what is exact, weights or log-odds ratios?" The weight *is* the log-odds ratio per unit; the exact contribution is weight × (value − mean) in log-odds |
| SHAP also computed in log-odds | In log-odds SHAP equals the exact contributions to four decimals (top-5 overlap 1.00). The 0.82 overlap in probability units comes only from the sigmoid |
| LIME with and without bins | Feedback on the BMI bins. Result: without bins LIME is not better (overlap 0.45 vs 0.54), so the bins are not the main problem |
| All percentages 0-100 / percentage points | Per's feedback |
| Points table greys out unreliable features | Heavy drinking appeared as a green, "good" bar |
| Patient note gives the real share with (pre)diabetes | With real prevalence, "half of them" is no longer true |

**Technical audit (new in Project 2): what has the network learned?**

| Decision | Why |
|---|---|
| Two of the three methods: latent space + concepts. CKA dropped | Assignment needs at least two. Week 10 was cancelled, the week 11 exercises have no CKA, and for this network CKA would only repeat that the NN orders persons like the LR (first run 0.92) |
| Latent space on 4000 random test persons with real class shares | Enough for t-SNE/UMAP, keeps prediabetes at its real share (~70 persons) |
| PCA, random projection, t-SNE and UMAP on inputs and both hidden layers, with trustworthiness and a perplexity / n_neighbors / seed check | Week 9 limits: linear methods can overlap groups; t-SNE/UMAP distances and cluster sizes mean little and can create clusters from noise |
| Main figure: UMAP of the inputs next to UMAP of hidden layer 2 | One contrast shows what the network does with the answers. PCA would be simpler but projects the two low-risk groups on top of everyone else |
| Eight concepts defined from inputs (metabolic profile, poor health, cardiovascular history, healthy lifestyle, heavy drinker, young adult, low socio-economic status, barriers to care) | Groups a doctor or the patient organisation talks about; none is a single input |
| Probes with a shuffled-label baseline | Week 11 exercise 4: a probe accuracy only means something next to chance (~0.5) |
| TCAV in hidden layer 1, 20 CAVs per concept, 20 random CAVs as null, t-test | Kim et al.; layer 1 because the last layer's gradient is constant |
| Counterexamples = persons *without* the concept (departure from Kim et al.) | Random counterexamples contain concept members (6 % of random persons are heavy drinkers) and made the heavy-drinker CAV unstable (sd 0.26 vs 0.06) |
| CAV built with logistic regression, checked with all five week 11 methods | The result for uncertain concepts depends on the method (B.4) |
| Input-level checks next to TCAV | A CAV mixes the traits that come with a concept; giving persons the concept directly shows the model's real reaction |
| NMF with k = 8 on hidden layer 1, top 5 % per concept, five random restarts | Finds concepts we did not define (the assignment asks about "surprising concepts"); k = 8 because 7 of 8 concepts are stable |
| Ablation (our own addition, 12.4): move each member's coordinate along the CAV to the non-members' average, rest of the network unchanged | TCAV only gives the sign of a tiny change; ablation measures how many points of the score a concept carries. Average non-member instead of zero, because ReLU activations are never negative and zero is a value no real person has |
| Ablation on up to 1000 test persons *with* the concept (any label), CAVs as in TCAV, 10 runs, all five methods | Only 48 heavy drinkers and 35 young adults among the 2000 (pre)diabetes persons used for TCAV |
| Random-direction baseline for ablation + input reference (closest answer outside the concept, e.g. age 35-39) | The baseline shows which method can be trusted (PCA cannot: random directions change the risk as much); the input reference is a known true effect to compare against |
| Sizes read from the logistic-regression CAV | Closest to the input reference (young adult −1.2 vs −1.2, male +1.6 vs +2.7, no college −0.4 vs −0.2); mean difference removes everything that comes with the group (no college +10.6) |
| Deeper network as a challenge (our own addition, 12.5): depths 1-5 with halving widths + 4 × 128, three seeds each, same penalty and training | Tests whether depth helps (seed spread is the noise a difference must beat) and lets us follow concepts layer by layer; only the architecture changes |
| Deep network 128 + 64 + 32 + 16 | Continues the main funnel (first two layers comparable), gives three layers for TCAV; 5 layers end in 8 units with ~3 dead, 4 × 128 ends with 31 dead |
| Per layer: probe, TCAV (layers 1-3) and ablation (layers 1-4), logistic-regression and mean-difference CAVs only | TCAV impossible in the last hidden layer; PCA failed the ablation baseline; five methods × four layers would be 200 numbers without a new question |
| Probes and TCAV for sex, income and education in the network trained without them | Tests whether removing sensitive features makes the network "blind" |

### B.3 Results to remember

- AUC LR 0.815, NN 0.822; average precision 0.428 vs 0.455. Calibration good for both.
- Of 100 flagged, only ~34 have (pre)diabetes (low precision is the price of screening at 16 % prevalence).
- Prediabetes caught 65 (LR) / 68 (NN) per 100; diabetes 77 / 80.
- Young people mostly missed (18-24: 91-96 % missed; 25-44: 46-63 %).
- Latent space: hidden layer 2 is close to one risk direction (PCA keeps 67 % of its variance in two axes, 26 % for the
  inputs). The network adds little over the LR.
- TCAV: metabolic profile, poor health, cardiovascular history 1.00 with all five methods; low socio-economic status
  0.98, barriers to care 0.95; young adult 0.18.
- Ablation (layer 1, LR CAV): metabolic profile carries 17.8 points of its members' risk (of 42.4 %), poor health 11.6,
  cardiovascular 7.8; low income 3.4, low SES 2.9, male 1.6, barriers 1.5; healthy lifestyle +0.4, no college −0.4,
  heavy drinking −1.5 (input −5.2). Random baseline ≤ 0.1.
- Deeper network: AUC 0.820-0.822 for every depth (seed sd ≤ 0.0008); concepts fade layer by layer (sex probe 1.00 →
  0.70 in layer 4 → 0.51 in the output); TCAV with the LR CAV flips between layers (barriers 0.96 / 0.83 / 0.01).
- Network without sex/income/education still encodes them (low income probe 63 % vs 51 % shuffled, AUC 0.75) and uses
  the low-income direction (TCAV 0.92). Its scores per income group are nearly unchanged.

### B.4 What was odd

| Oddity | What we saw | Likely reason / explanation | Consequence |
|---|---|---|---|
| **Heavy drinking still protects** | Stopping heavy drinking +5 pp in both models (12 pp in Project 1); 54 % / 49 % of heavy drinkers with (pre)diabetes missed vs 21-23 %; own low-risk region in layer 2 of the latent space; TCAV 0.20 (but SVM CAV 0.87); input check: giving heavy drinking lowers risk 11 pp for 99.8 % | Reverse causation as in Project 1. The effect is smaller in pp because probabilities near 16 % move less than near 50 % | Retrain without it: AUC −0.001, missed heavy drinkers 28-30 %, remaining unsafe effects < 0.4 pp. Deploy the retrained model |
| **Cholesterol check has the largest weight of all** | LR weight 1.19 (odds × 3.3); persons without a check form their own low-risk region in the latent space (predicted 2.7 %, observed 3.0 %); NMF finds them as a concept without being asked; LIME ranks it 1-2, truth 7.8 | The label is "ever *told* by a doctor". Without a check (and often without a doctor) a person is less likely to be diagnosed, so the low risk is a *detection* effect, not health. LIME: samples features independently; a sampled "no" (rare, 4 %) is a large jump in log-odds, so the surrogate overweights it | Greyed out in the points table; never shown as a reason to the patient; SHAP, not LIME |
| **Health insurance raises the risk** | LR weight +0.084 | Same detection effect: insured people see doctors and get diagnosed | Mention with cholesterol check |
| **Fewer bad days raises the risk** | Mental and physical health days flagged unsafe in both models (LR weights −0.002 and −0.007) | General health and difficulty walking already carry the information; the remaining partial effect is slightly negative (collinearity). Small: < 0.4 pp | Never advise on them; they are health status, not habits |
| **Vegetables raise the risk in the network** | Advice-safety flag for the NN only | Small non-linear artefact | Not advised |
| **Smoking and fruit barely matter** | LR smoking weight 0.003; NMF finds a smoker / no-fruit / cost-barrier profile that lowers risk (TCAV 0.01, predicted 8.7 %); changing only smoking and fruit moves the risk < 0.2 pp | The low risk comes from what comes with the profile in the data, not the habits; the model gives no reward for quitting smoking or eating fruit | The patient view must not suggest changes the model does not reward |
| **Healthy lifestyle looks like it raises the risk** | TCAV 0.72 with the LR CAV, 0.00 with mean difference; input check −0.7 pp, rises for 30 % | A CAV follows everything that separates the groups; when the concept direction is nearly orthogonal to the gradient, a small rotation flips the score | Rule: a concept counts as used only when the CAV methods agree or an input check confirms |
| **The CAV method changes the TCAV result** | Clear concepts: all five methods agree. Heavy drinker, healthy lifestyle, young adult: methods disagree | Almost all persons' gradients point the same way, so a TCAV score is close to the sign of one dot product (random directions also land near 0 or 100) | Same rule as above |
| **TCAV says "used", ablation says "barely"** | Low SES TCAV 0.98 and barriers 0.95, almost like the metabolic profile (1.00); ablation 2.9 and 1.5 points against 17.8 | TCAV counts how many persons' risk rises, not by how much | Sizes from ablation, direction from TCAV + input checks |
| **The SVM CAV for heavy drinking** | TCAV 0.87 (raises), but removing that direction changes the risk by only 0.3 points | The network hardly responds to that direction, so the TCAV score is the sign of an almost zero effect | Heavy drinking settled: LR CAV, mean difference and input all say it lowers the risk |
| **Deeper layers scramble concepts** | Cardiovascular history ablation +9.7 in layer 1, −4.0 in layer 4; mean-difference random baseline grows from 4.9 to 29.9 points | In a 16-unit layer almost every direction overlaps the risk direction | Audit in hidden layer 1; keep two layers |
| **"No health insurance" is encoded but not used** | NMF concept, TCAV 0.43 (not significant), while our "barriers to care" scored 0.95 | The signal probably comes from skipping the doctor because of cost | |
| **Higher AUC but more often wrong** | NN AUC 0.822 > LR 0.815, but at the chosen thresholds the LR is right for 1863 persons where the NN is wrong, the NN for 1622 (McNemar p = 4.8e-5) | Different operating points: the NN catches more cases with more false alarms, and with 84 % negatives false alarms dominate the count | Compare models at the same recall (an argument for the recall-80 % rule) |
| **High earners and graduates missed most** | Missed: income > $75k 37-41 %, college graduates 31-36 %; lowest income 11-12 % | Low prevalence in these groups pushes their scores down | Report errors per group to the data scientist |
| **Implausible BMI values** | The most confident false alarms have BMI 87-98 | Probably data entry errors in the survey | Data-quality point; not cleaned (would need a rule we cannot justify from the data alone) |

### B.5 Challenges

1. **Imbalance (15.8 % positive).** 0.5 is useless (recall 18 %), accuracy is meaningless. Solved with threshold
   choice on validation and imbalance-aware metrics, not by changing the data.
2. **A very small class (prediabetes 1.8 %).** Merged into the target, but reported separately because it is caught
   least and it is the group where screening helps most.
3. **Keeping the score a real risk.** Class weights and resampling would break calibration, which the patient and
   director figures depend on.
4. **Low precision.** Two of three flagged persons do not have (pre)diabetes: acceptable for screening (the follow-up
   is a blood test), not for diagnosis.
5. **The same wrong-direction features as in Project 1**, now with a clearer explanation (detection effect and reverse
   causation in a survey label).
6. **Auditing what a network has learned without fooling ourselves.** t-SNE/UMAP can invent clusters; TCAV depends
   on counterexamples and CAV method; random CAVs give 0 or 100; probes show presence, not use. Every result needed a
   second check (trustworthiness, settings/seeds, five CAV methods, input-level checks, NMF stability).
7. **Sensitive attributes through proxies.** Removing them does not make the network blind; the decision moves to
   how the features are shown (summed background bar, no advice, errors per group).
8. **Size.** 253,680 persons: SHAP for the network takes about a minute for 1000 persons, so explanations and the
   audit use samples (1000, 100, 4000, 20,000).
9. **The network adds little.** AUC +0.007, one risk direction in the last layer, and less often right than the LR at
   the chosen thresholds. Which model to recommend is part of the deployment decision.
10. **Course changes.** Week 10 cancelled, so CKA dropped.
11. **Our "challenge": beyond the course methods.** The teacher asked us to challenge ourselves somewhere. We chose
    the network: concept ablation (a causal test inside the network, which needed its own baseline and reference
    because nothing in the course defines how to remove a concept) and a deeper network followed layer by layer.
    The hard part was not running them but deciding which numbers to believe (PCA and mean difference fail the
    random baseline in deep, narrow layers).

### B.6 Still open
- Threshold rule (Youden's J vs recall 80 %).
- Which model to recommend for deployment (the retrained model without heavy drinking in any case).
- Validation summary table (notebook 7.7) is empty; recommendation text (11) not written.
- USMs for all three stakeholders, Before/After designs, think-aloud study, report, presentations.

---

## Part C. Project 1 → Project 2 at a glance

| | Project 1 | Project 2 |
|---|---|---|
| Data | 70,692, 50/50 | 253,680, 15.8 % positive (prediabetes 1.8 %) |
| Score | Not a real risk (average person 48) | Calibrated risk (average ~16) |
| Threshold | 0.5 | Chosen on validation (rule open) |
| Network | 1 × 32, alpha 1e-4 | 128 + 64, alpha 0.01 |
| AUC LR / NN | 0.823 / 0.828 | 0.815 / 0.822 |
| SHAP vs exact | Probability units, top-5 Jaccard 0.97 | Also log-odds: identical (1.00) |
| LIME | Quartile bins only | With and without bins |
| Heavy drinking effect | +12 pp when stopping | +5 pp when stopping |
| Audit of the network | None | Latent space, probes, CAV/TCAV (5 methods), NMF, concept ablation, deeper network per layer, hidden sensitive attributes |
