# Financial Transaction Irregularity Detection with PySpark

**Course Project:** Phase 3 – Data Intensive Computing  
**Authors:** Sai Sri Kolanu (50594437), Jyothsna Devi Goru (50560456), Hamsika Rajeshwar Rao (50613199)  
**University at Buffalo**  

---

## 📖 Overview
This project uses **PySpark** to analyze irregularities in global financial transactions by processing the **Big Black Money Dataset**.  
The workflow includes:
- Data preprocessing (cleaning, feature engineering, outlier removal, encoding, normalization).  
- Distributed transformations and DAG visualizations for execution analysis.  
- Machine learning models with **PySpark MLlib** for detecting suspicious transactions.  

---

## 📊 Dataset
- **Source:** [Global Black Money Transactions Dataset (Kaggle)](https://www.kaggle.com/datasets/waqi786/global-black-money-transactions-dataset)  
- **Size:** 10,000 rows × 14 columns  
- **Key Columns:** Transaction ID, Amount (USD), Countries, Industry, Financial Institution, Risk Score, Shell Companies.  
- **Goal:** Identify irregular transactions linked to money laundering and financial crimes.  

---

## ⚙️ Data Processing Pipeline
1. **Convert date → timestamp** for temporal analysis.  
2. **Handle missing values** (mean imputation).  
3. **Remove duplicates** (based on Transaction ID).  
4. **Remove outliers** (IQR filtering on Amount).  
5. **Risk categorization** (High/Medium/Low from risk score).  
6. **One-hot encoding** of categorical variables.  
7. **Normalize Amount (USD)** with MinMaxScaler.  
8. **Aggregate transactions per person** (total & count).  
9. **Window functions** to compute rolling averages for trends.  

---

## 🤖 Machine Learning Models (PySpark MLlib)
- **Naive Bayes** – Accuracy: ~92%  
- **Logistic Regression** – Accuracy: ~94.9%, Precision/Recall ~94.9%  
- **KNN Classifier** – Accuracy: ~92%, F1 ~95.6%  
- **Support Vector Machine (SVM)** – Accuracy: ~92%  
- **Stochastic Gradient Descent (SGD)** – Experimental  
- **Multilayer Perceptron (MLP)** – Accuracy: ~79%, but failed on positive class due to imbalance  

Metrics: **Accuracy, Precision, Recall, F1 Score, Execution Time**  

---

## 📈 Results
- **Best Model:** Logistic Regression (Accuracy ~94.9%, balanced Precision/Recall).  
- **KNN** performed well on positives, slightly slower execution.  
- **MLP** underperformed due to class imbalance.  
- **DAG Visualizations** provided insights into Spark’s execution stages and optimizations.  

---

## 🛠️ Tech Stack
- **Python 3.x**
- **Apache Spark (PySpark)**
- **MLlib**
- **Matplotlib, Seaborn** (for plots)

---

## ▶️ How to Run
1. Start Spark environment and load dataset.  
2. Submit Spark job:
   ```bash
   spark-submit preprocessing.py
   spark-submit model_training.py
