# Online Shoppers Purchasing Intention - Classification & Streamlit Deployment

**M.Tech (AIML) - Machine Learning - Assignment 2**

---

## a. Problem statement

An e-commerce site records, for every browsing session, which page types the visitor viewed, how long they stayed, Google-Analytics engagement metrics (bounce rate, exit rate, page value) and context such as month, visitor type and traffic source. Only about 1 in 6 sessions ends in a purchase.

**Goal:** build and compare binary classifiers that predict, from the session features, whether a session will end in a purchase (`Revenue = 1`) or not (`Revenue = 0`). The five required models are Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes and Support Vector Machine. They are evaluated with Accuracy, AUC, Precision, Recall, F1 and MCC, and the best one is identified. The trained models are served through an interactive Streamlit app that evaluates them on uploaded test data without retraining.

Because the classes are imbalanced (≈ 84 : 16), a model that always predicts "no purchase" already reaches ≈ 84% accuracy. Model quality is therefore judged mainly on **MCC, F1 and AUC**, not on accuracy alone.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository, dataset #468 - <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset> |
| Original reference | Sakar, C.O., Polat, S.O., Katircioglu, M., Kastro, Y. (2018), *Neural Computing and Applications* |
| Instances | 12,330 sessions (12,205 after removing 125 exact duplicate rows) |
| Features | 17 input features + 1 target (`Revenue`) |
| Task | Binary classification |
| Class balance | `Revenue = 0`: 84.4%, `Revenue = 1`: 15.6% (after de-duplication) |
| Missing values | None |

Each session belongs to a different user over a 1-year period, which avoids bias towards a particular campaign, day or user.

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | Administrative | int | Number of administrative (account) pages visited |
| 2 | Administrative_Duration | float | Total time (s) spent on administrative pages |
| 3 | Informational | int | Number of informational pages visited |
| 4 | Informational_Duration | float | Total time (s) on informational pages |
| 5 | ProductRelated | int | Number of product-related pages visited |
| 6 | ProductRelated_Duration | float | Total time (s) on product-related pages |
| 7 | BounceRates | float | Avg. bounce rate of the pages visited (Google Analytics) |
| 8 | ExitRates | float | Avg. exit rate of the pages visited (Google Analytics) |
| 9 | PageValues | float | Avg. page value of the pages visited before a transaction |
| 10 | SpecialDay | float (0-1) | Closeness of the visit date to a special day (e.g. Mother's Day, Valentine's Day) |
| 11 | Month | categorical | Month of the visit (10 distinct values; Jan and Apr are absent) |
| 12 | OperatingSystems | categorical (code 1-8) | Operating system of the visitor |
| 13 | Browser | categorical (code 1-13) | Browser of the visitor |
| 14 | Region | categorical (code 1-9) | Geographic region of the visitor |
| 15 | TrafficType | categorical (code 1-20) | Traffic source (banner, SMS, direct, ...) |
| 16 | VisitorType | categorical | New_Visitor / Returning_Visitor / Other |
| 17 | Weekend | boolean | Whether the visit was on a weekend |
| - | **Revenue** (target) | boolean | Whether the session ended in a purchase |

**Preprocessing.** The same steps are applied to every model and stored inside each saved pipeline:
1. 125 exact duplicate rows are removed **before** splitting, so identical rows cannot appear in both train and test.
2. The data is split 80/20 with a stratified split (`random_state = 42`). That gives 9,764 training rows and 2,441 test rows; the test rows are saved as `test_data.csv`.
3. `log1p` and then standardisation are applied to the 7 heavily right-skewed counts, durations and `PageValues`. For example, `ProductRelated_Duration` has a median of ≈ 599 s but a maximum of ≈ 64,000 s.
4. Standardisation is applied to `BounceRates`, `ExitRates` and `SpecialDay`.
5. One-hot encoding (`handle_unknown='ignore'`) is applied to `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region` and `TrafficType`. The integer codes are category IDs, not magnitudes.
6. `Weekend` is mapped to 0/1. The final design matrix has 73 columns.

## c. GitHub Repository Link

**Repository:** <https://github.com/Ajayakumar-Vijayakumar-2026aj05144/ml-assignment-2>

**Live Streamlit App:** <https://ml-assignment-2-tac5ifdmmbnbuda7zuenza.streamlit.app/>

Repository structure:

```
ml-assignment-2/
├── app.py                  # Streamlit web application
├── train_models.py         # Training, CV tuning, evaluation, saving of all 5 models
├── verify_models.py        # Reloads saved models and re-checks metrics (no fitting)
├── preprocessing.py        # Shared feature lists, type cleaning, ColumnTransformer
├── requirements.txt        # Pinned dependencies
├── README.md
├── test_data.csv           # Held-out 20% test set (17 features + Revenue)
├── data/
│   └── online_shoppers_intention.csv   # Raw UCI dataset
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    ├── metrics.csv             # Test-set comparison table
    ├── best_params.json        # Tuned hyperparameters + CV F1
    ├── feature_schema.json     # Feature list, model files, versions
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── training_log.txt
```

## d. Models used

Each model is an sklearn `Pipeline` made of the preprocessing above followed by the classifier. Hyperparameters were chosen with `GridSearchCV` (5-fold stratified CV, scoring = F1) **on the training set only**. Each tuned model was then evaluated **once** on the held-out test set (2,441 rows, 382 positives). The positive class for Precision, Recall and F1 is `Revenue = 1`. AUC uses `predict_proba`, or `decision_function` for the SVM.

| ML Model | Tuned hyperparameters | CV F1 (train) |
|---|---|---|
| Logistic Regression | C = 0.003, class_weight = balanced (lbfgs) | 0.6673 |
| Decision Tree | criterion = entropy, max_depth = 4, min_samples_leaf = 200, class_weight = None | 0.6612 |
| KNN | n_neighbors = 21, weights = distance, p = 2 (Euclidean) | 0.6090 |
| Naive Bayes (Gaussian) | var_smoothing = 1.0 | 0.5926 |
| Support Vector Machine | RBF kernel, C = 3, gamma = 0.003, class_weight = balanced | 0.6675 |

### Comparison table (held-out test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8808 | 0.9206 | 0.5880 | 0.7958 | 0.6763 | 0.6157 |
| Decision Tree | **0.8976** | 0.9244 | 0.6854 | 0.6387 | 0.6612 | 0.6015 |
| KNN | 0.8927 | 0.9122 | **0.7027** | 0.5445 | 0.6136 | 0.5585 |
| Naive Bayes | 0.8570 | 0.8720 | 0.5324 | 0.7094 | 0.6083 | 0.5311 |
| Support Vector Machine | 0.8828 | **0.9259** | 0.5934 | **0.7984** | **0.6808** | **0.6210** |

Bold = best value in the column. Reference point: always predicting "no purchase" gives Accuracy = 0.8435 and MCC = 0.

Confusion matrices on the test set (TN / FP / FN / TP):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 1846 | 213 | 78 | 304 |
| Decision Tree | 1947 | 112 | 138 | 244 |
| KNN | 1971 | 88 | 174 | 208 |
| Naive Bayes | 1821 | 238 | 111 | 271 |
| Support Vector Machine | 1850 | 209 | 77 | 305 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model: MCC 0.6157 and F1 0.6763, essentially tied with the SVM. CV picked strong regularisation (C = 0.003) and `class_weight = balanced`. The class weighting moves the decision boundary towards the minority class, giving high Recall (0.7958, only 78 buyers missed) at the cost of lower Precision (0.5880, 213 false alarms). The largest coefficient by far is `PageValues` (+1.16), followed by `Month_Nov` (+0.47). `ExitRates`, `BounceRates` and `Month_May`/`Month_Mar` push towards "no purchase". After the log transform the problem is close to linearly separable, which is why such a simple model competes with the kernel SVM. It is also fast and interpretable. |
| Decision Tree | Highest Accuracy (0.8976) and second-best AUC (0.9244), with the most balanced Precision/Recall trade-off (0.6854 / 0.6387). CV chose a heavily pruned tree (depth 4, ≥ 200 samples per leaf), which shows that deeper trees over-fit this data. `PageValues` alone accounts for ≈ 77% of the total feature importance, followed by `Month_Nov` (8%) and `BounceRates` (5%). The tree does not use class weighting, so it misses more buyers than LR/SVM (FN = 138 vs ≈ 78). That lowers its F1 (0.6612) and MCC (0.6015) despite the higher accuracy. This is a clear example of accuracy being misleading on imbalanced data. |
| KNN | Highest Precision (0.7027, only 88 false alarms) but the lowest Recall of all models (0.5445, missing 174 of 382 buyers). F1 (0.6136) and MCC (0.5585) are therefore the second weakest. With k = 21 the majority "no purchase" class dominates most neighbourhoods. The Euclidean distance is computed over 73 dimensions, about 62 of them sparse one-hot columns. Those columns dilute the few highly informative features such as `PageValues`, so the neighbourhoods are less class-pure. KNN also cannot use class weights. It is a lazy learner: every prediction computes distances to all 9,764 stored training rows, so `knn.pkl` is the largest artifact (≈ 5.8 MB). |
| Naive Bayes | Weakest model on every metric except Recall: AUC 0.8720, F1 0.6083, MCC 0.5311, Precision 0.5324. Gaussian NB assumes conditionally independent, normally distributed features. Both assumptions are broken here: page counts and their durations are strongly correlated, as are `BounceRates` and `ExitRates`, and 63 of the 73 columns are binary (62 one-hot indicators plus `Weekend`). The tuned `var_smoothing = 1.0` is very large. It adds a variance floor that makes the model less over-confident about the binary columns. With weaker smoothing (`1e-1`, on a narrower grid) accuracy fell to 0.7513 and MCC to 0.4533 because there were many false positives. NB is still useful as a very fast, low-variance baseline. |
| Support Vector Machine | Best overall: highest MCC (0.6210), F1 (0.6808), AUC (0.9259) and Recall (0.7984, finding 305 of 382 buyers). The RBF kernel with a small gamma (0.003) gives a smooth, almost linear boundary, which explains why its results are so close to Logistic Regression. `class_weight = balanced` again trades some Precision (0.5934) for Recall. The drawbacks are cost and interpretability: tuning took ≈ 260 s versus ≈ 10 s for LR, and it has no native probability output (AUC is computed from the decision function). |
| Overall, Winner for your dataset? | **Support Vector Machine (RBF)**, chosen on MCC, the most reliable single metric for this imbalanced problem. It also has the best F1, AUC and Recall. However, its lead over Logistic Regression is very small (MCC +0.005, F1 +0.005: one more true positive and four fewer false positives on 2,441 test rows). That difference is within test-set noise. In practice, Logistic Regression is an equally good and cheaper, more interpretable choice. If false alarms are expensive and Precision/Accuracy matter more, the shallow Decision Tree is the better option. Across all models `PageValues` is the dominant predictor of purchase. |

---

## Streamlit app features

The live app is at the link in section c. It provides:
- **CSV upload** of test data (17 features + `Revenue`). The columns and types are validated, and the bundled `test_data.csv` can be downloaded or used directly.
- **Model selection dropdown** covering all 5 saved classifiers.
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC as metric cards.
- **Confusion matrix** heatmap, **ROC curve** and full **classification report**.
- A **comparison tab** that scores all 5 models on the uploaded data, with a table, best-by-MCC highlight and grouped bar chart.
- Row-level **predictions**, with a filter for misclassified rows and a CSV download.
- A **training details** tab with the tuned hyperparameters and the reference metrics.

The app only loads `model/*.pkl` and calls `predict` / `predict_proba` / `decision_function`. **It never fits a model on the uploaded data.** `verify_models.py` confirms that the reloaded models reproduce `model/metrics.csv` exactly.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py      # optional - re-creates model/*.pkl, metrics and test_data.csv
python verify_models.py     # optional - checks saved models reproduce the metrics
streamlit run app.py
```

Environment used: Python 3.12, scikit-learn 1.9.0. The pickles are tied to this scikit-learn version, which is why `requirements.txt` pins it. On Streamlit Community Cloud, select **Python 3.12** under *Advanced settings* when deploying.
