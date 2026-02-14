# Customer Personality Analysis – Clustering

This project performs **customer segmentation** using unsupervised machine learning techniques on the *Customer Personality Analysis* dataset.  
The goal is to identify meaningful customer groups based on demographic and purchasing behavior to support data-driven marketing strategies.


📓 **Kaggle Notebook:**  
https://www.kaggle.com/code/leyuzakoksoken/customer-personality-analysis-clustering

---

## 📌 Project Overview

Customer segmentation is a key task in marketing analytics. In this project, clustering algorithms are applied to group customers with similar characteristics.  
The analysis follows an **end-to-end data science workflow**, including:

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Clustering with multiple algorithms
- Model evaluation and comparison
- Deployment as an interactive web application

---

## 📊 Dataset

- **Name:** Customer Personality Analysis  
- **Source:** Kaggle  
- **Records:** 2,240 customers  
- **Features:** 29 (demographics, income, purchase behavior, campaign response)

The dataset contains information such as:
- Age and education
- Household composition
- Income level
- Product spending habits
- Customer recency and engagement

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Removal of non-informative identifiers (ID)
- Transformation of `Dt_Customer` into customer tenure (days)
- One-hot encoding of categorical variables
- Median imputation for missing values
- Feature scaling using `StandardScaler`

These steps ensure that clustering is performed on clean and comparable features.

---

## 🤖 Models Used

The following clustering algorithms were trained and evaluated:

- **KMeans** (final selected model)
- **Agglomerative Clustering**
- **DBSCAN**

### Model Evaluation Metrics
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score

📌 **Result:**  
KMeans provided the most stable and interpretable clusters for this dataset and was selected as the final model.

---

## 📐 Elbow Method

The Elbow Method was used to determine the optimal number of clusters (`k`).  
Based on inertia values, **k = 4** was selected as the optimal configuration.

---

## 📈 Visualization

- Feature distributions during EDA
- PCA-based 2D visualization of clusters
- Cluster distribution bar charts
- Segment-level statistical summaries

---

## 🌐 Web Application (Streamlit)

The trained model was deployed as an interactive **Streamlit dashboard** and hosted on **HuggingFace Spaces**.

### App Features
- CSV file upload
- Automatic preprocessing
- Cluster prediction
- Segment naming
- PCA visualization
- Downloadable segmented dataset

🔗 **Live App:**  
https://huggingface.co/spaces/leyuzak/Customer-Personality-Analysis-Clustering

