# Loan Approval Prediction

Predicts loan approval status from applicant demographic and financial data. Covers EDA, feature engineering, and comparison of four classification models — with and without class-balancing via SMOTE.

## Dataset
- **Source:** Loan Prediction dataset (614 rows, 13 features)
- **Target:** `Loan_Status` (Approved/Rejected)

## Workflow
1. **EDA** — distribution and count analysis on Gender, Married, Dependents, Self_Employed, Credit_History; approval-rate breakdowns by applicant attributes; correlation heatmap.
2. **Missing values** — categorical columns imputed with mode, numeric columns (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`) imputed with mean.
3. **Feature engineering** — created `Total_Income` (Applicant + Coapplicant income); applied log transformation to Income and Loan Amount to correct right-skew.
4. **Encoding** — label-encoded categorical features (Gender, Married, Education, Self_Employed, Property_Area, Loan_Status).
5. **Modeling** — trained Logistic Regression, Decision Tree, Random Forest, and KNN; evaluated with train/test split and 5-fold cross-validation.
6. **Class balancing** — re-ran all four models after applying SMOTE to address target imbalance.

## Results

**Test accuracy — original (imbalanced) data:**
| Model | Test Accuracy | 5-Fold CV Mean |
|---|---|---|
| Logistic Regression | 77.3% | 81.1% |
| Decision Tree | 72.1% | 70.9% |
| Random Forest | **79.2%** | — |
| KNN (k=3) | 69.5% | 73.1% |

**Test accuracy — after SMOTE:**
| Model | Test Accuracy |
|---|---|
| Logistic Regression | 68.2% |
| Decision Tree | 74.9% |
| Random Forest | 76.8% |
| KNN (k=3) | 69.7% |

**Best model: Random Forest on the original, imbalanced dataset (79.2% test accuracy).** SMOTE reduced accuracy across every model in this run rather than improving it — likely because the resampled synthetic minority points didn't generalize well to the small (614-row) dataset. Worth revisiting with a different balancing ratio or precision/recall-focused tuning rather than accuracy alone.

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Imbalanced-learn (SMOTE)

## Files
- `Loan_Approval_Prediction.ipynb` — full pipeline: EDA → feature engineering → encoding → 4-model comparison → SMOTE re-evaluation
- `README.md`

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook Loan_Approval_Prediction.ipynb
```
## Author
Neha Bhan — Data Analyst | Python · SQL · Power BI
[LinkedIn](https://linkedin.com/in/neha-bhan-analyst) | [GitHub](https://github.com/nehakavi30-pixel)
