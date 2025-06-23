# 🏦 Loan Eligibility Prediction Model (Data Science Project)

This project focuses on analyzing and understanding a loan dataset through data cleaning, exploratory data analysis (EDA), and building a machine learning model to predict loan eligibility.

---

## 📁 Project Structure

- `loan_eda_cleaning.ipynb` – Jupyter Notebook with cleaning, EDA, model building, and evaluation.
- `loan_dataset.csv` – Loan dataset used in the project.
- `README.md` – Project documentation (this file).

---

## 📊 Dataset Features

| Column Name         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Loan_ID**          | Unique identifier for each loan application                                |
| **Gender**           | Gender of the applicant (Male/Female)                                      |
| **Married**          | Marital status of the applicant (Yes/No)                                   |
| **Dependents**       | Number of dependents (0, 1, 2, 3+)                                          |
| **Education**        | Education level (Graduate/Not Graduate)                                     |
| **Self_Employed**    | Employment status (Yes/No)                                                  |
| **ApplicantIncome**  | Monthly income of the applicant                                             |
| **CoapplicantIncome**| Monthly income of the co-applicant                                          |
| **LoanAmount**       | Loan amount requested (in thousands)                                       |
| **Loan_Amount_Term** | Term of the loan (in months, usually 360)                                  |
| **Credit_History**   | Credit history (1 = good, 0 = bad, may contain missing values)              |
| **Property_Area**    | Area type where the property is located (Urban/Semiurban/Rural)            |
| **Loan_Status**      | Target variable (Y = loan approved, N = not approved)                      |

---

## 🧹 Data Cleaning Performed

- Treated missing values in `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, etc.
- Converted categorical columns using Label Encoding or One-Hot Encoding
- Removed duplicates and irrelevant columns
- Handled outliers in `ApplicantIncome`, `LoanAmount`

---

## 🔍 Exploratory Data Analysis (EDA)

- Distribution of loan approvals vs rejections
- Average loan amount by education, employment status
- Loan approval rate by credit history
- Analysis by property area, dependents, and gender
- Correlation between numeric variable

## 🤖 Machine Learning Model

A classification model was built to predict whether a loan will be approved or not.

### ✅ Model Used:
- Logistic Regression

### ⚙️ Evaluation Metrics:
- **Accuracy Score**: Evaluated model performance
- **Confusion Matrix**: Showed prediction breakdown

```python
from sklearn.metrics import accuracy_score, confusion_matrix

# Example usage:
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
