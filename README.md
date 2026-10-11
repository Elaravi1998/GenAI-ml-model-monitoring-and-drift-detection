# 📊 ML Model Monitoring & Drift Detection

A practical MLOps project for detecting data drift, monitoring model predictions, validating data quality, and tracking machine-learning performance.

## ✨ Features
- 🧪 Train a baseline Logistic Regression model
- 📦 Simulate stable and drifted production batches
- 🧹 Check missing values, duplicate rows, and schema changes
- 📈 Detect numeric drift with the Kolmogorov–Smirnov (KS) test
- 🧮 Detect categorical drift with a chi-square test
- 🎯 Monitor prediction rates and probability distributions
- 📉 Evaluate accuracy, precision, recall, F1, and ROC-AUC when labels are available
- 🚨 Generate configurable alert summaries
- 📤 Export monitoring reports to CSV and JSON

## 🧰 Technology Stack
Python, NumPy, pandas, scikit-learn, SciPy, Matplotlib, Jupyter Notebook, Google Colab.

## 🗂️ Repository Structure
```text
ml-model-monitoring-and-drift-detection/
├── ML_Model_Monitoring_and_Drift_Detection.ipynb
├── README.md
├── monitoring_summary.csv
└── monitoring_summary.json
```
The CSV and JSON files are generated when the notebook's export cell runs.

## 🚀 Installation

Clone your repository:
```bash
git clone https://github.com/YOUR_USERNAME/ml-model-monitoring-and-drift-detection.git
cd ml-model-monitoring-and-drift-detection
python -m venv .venv
```

Activate on Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and launch Jupyter:
```bash
pip install numpy pandas scikit-learn scipy matplotlib jupyter
jupyter notebook
```

Open `ML_Model_Monitoring_and_Drift_Detection.ipynb` and run the cells in order. You can also upload the notebook to [Google Colab](https://colab.research.google.com/).

## 🔍 Workflow
1. Generate reference classification data and train a baseline model.
2. Create stable, moderately drifted, and strongly drifted production batches.
3. Check schema, duplicate rows, and missing-value percentages.
4. Compare numeric feature distributions using KS tests.
5. Compare categorical frequencies using a chi-square test.
6. Monitor prediction rates and probability distributions.
7. Evaluate model performance when ground-truth labels are available.
8. Combine findings into alerts and export reports.

## 📐 Drift Detection Methods
- **KS test:** compares two numeric sample distributions.
- **Chi-square test:** compares categorical counts with expected reference proportions.
- **Future options:** Population Stability Index (PSI), Jensen–Shannon divergence, and Wasserstein distance.

The notebook uses a simple multiple-testing correction for numeric features. P-values should be interpreted alongside effect sizes, sample sizes, domain thresholds, and business impact.

## 📊 Metrics
| Metric | Purpose |
|---|---|
| Missing-value rate | Data quality |
| KS statistic and p-value | Numeric feature drift |
| Chi-square statistic and p-value | Categorical distribution drift |
| Positive prediction rate | Output behavior |
| Accuracy | Overall classification correctness |
| Precision | Reliability of positive predictions |
| Recall | Positive cases detected |
| F1 score | Balance of precision and recall |
| ROC-AUC | Ranking performance |

## 🚨 Alerts and Reports
The example alert logic combines numeric drift flags, categorical drift, high missingness, and an F1 decrease when labels are available. It exports `monitoring_summary.csv` and `monitoring_summary.json`.

## ⚠️ Limitations
- The dataset is synthetic and does not represent a real business domain.
- Data drift does not automatically mean model performance has declined.
- Prediction drift cannot establish correctness without labels.
- Concept drift generally requires labels, domain feedback, or other evidence about feature-target relationships.
- Example alert thresholds must be calibrated before production use.
- This notebook is an offline prototype, not a continuously running monitoring service.
- Do not trigger model retraining from one unreviewed statistical test.

## 🛣️ Roadmap
- [ ] Add PSI and Jensen–Shannon divergence
- [ ] Build a Streamlit dashboard or Evidently AI report
- [ ] Track model versions with MLflow
- [ ] Schedule monitoring with GitHub Actions or a cloud scheduler
- [ ] Store historical metrics in a database
- [ ] Add email/Slack alerts and alert deduplication
- [ ] Simulate delayed labels and investigate concept drift
- [ ] Add unit tests and backtest alert thresholds

## 💼 Resume-Ready Description
Built a Python ML monitoring and drift detection workflow to track data quality, numeric and categorical feature shifts, prediction behavior, and model performance across simulated production batches. Implemented KS and chi-square statistical tests, configurable alert summaries, visual diagnostics, and CSV/JSON reporting using pandas, SciPy, scikit-learn, and Matplotlib.

## 🏷️ GitHub Topics
`machine-learning` `mlops` `model-monitoring` `data-drift` `drift-detection` `python` `scikit-learn` `data-quality` `model-performance` `jupyter-notebook`

## 📜 License
Consider adding an MIT License if you want others to reuse and contribute to the project.
