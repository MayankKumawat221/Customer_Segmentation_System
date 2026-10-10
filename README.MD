# 🛍️ Customer Segmentation

## 📌 Project Overview

This project is a Customer Segmentation system developed using Python and Machine Learning.

It groups mall customers into different segments based on their age, annual income, and spending score. These segments help businesses understand customer behavior and develop targeted marketing strategies.

## 🎯 Project Objectives

- Analyze customer data.
- Explore customer demographics and spending behavior.
- Identify customer groups with similar characteristics.
- Visualize customer clusters.
- Evaluate clustering performance.
- Predict segments for new customers.
- Save the final results and trained model.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- CSV

## 📂 Dataset

This project uses the Mall Customer Segmentation dataset.

**Dataset File:** `Mall_Customers.csv`

The dataset contains 200 customer records and 5 columns.

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier assigned to each customer |
| `Gender` | Gender of the customer |
| `Age` | Age of the customer |
| `Annual Income` | Annual income of the customer in thousands |
| `Spending Score` | Score from 1 to 100 representing customer spending behavior |

## 🔍 Project Workflow

### 1. Data Loading and Inspection

Loaded the dataset and inspected its first five rows, dimensions, and columns.

### 2. Data Cleaning

Checked for duplicate records and examined the dataset for missing values.

### 3. Exploratory Data Analysis (EDA)

Analyzed descriptive statistics and compared average annual income and spending scores across genders.

### 4. Feature Selection

Removed `CustomerID` and excluded `Gender` based on the observed similarities in average income and spending scores.

Selected features:
- Age
- Annual Income
- Spending Score

### 5. Feature Scaling

Standardized the selected features to ensure that differences in numerical scales would not disproportionately influence clustering.

### 6. Optimal Cluster Selection

Used the Elbow Method to analyze inertia and selected 6 clusters for the final model.

### 7. Model Training

Trained the clustering model using 6 clusters, a random state of 42, and 5 initializations.

### 8. Customer Cluster Assignment

Assigned each customer to a cluster and stored the cluster identifiers in the `labels` column.

### 9. Cluster Visualization

Created a scatter plot to visualize customer clusters based on annual income and spending score.

### 10. Customer Segment Interpretation

Calculated the average age, annual income, and spending score for each cluster and assigned descriptive segment names.

### 11. Model Evaluation

Evaluated clustering performance using the Silhouette Score.

**Silhouette Score:** Approximately 0.4284

### 12. Exporting Results

Saved the final customer segmentation results to `Customer_Segmentation.csv`.

### 13. Saving the Model and Scaler

Saved the trained clustering model and fitted scaler using Joblib.

### 14. Testing New Customers

Created sample data for three new customers, predicted their cluster assignments, and mapped the results to customer segment names.

## 🤖 Clustering Approach

This project uses an unsupervised machine learning approach to group customers with similar characteristics.

The model uses customer age, annual income, and spending score to identify six clusters.

The Silhouette Score of approximately 0.4284 indicates some separation between clusters, although further improvement may be possible.

## 📊 Customer Segments

| Cluster | Customer Segment |
|---|---|
| 0 | Older Medium-Income, Medium-Spending Customers |
| 1 | Young Medium-Income, Medium-Spending Customers |
| 2 | Middle-Aged High-Income, Low-Spending Customers |
| 3 | Young High-Income, High-Spending Customers |
| 4 | Young Low-Income, High-Spending Customers |
| 5 | Middle-Aged Low-Income, Low-Spending Customers |

These segment names are descriptive interpretations based on the average characteristics of each cluster.

## 📁 Project Structure

    Customer_Segmentation/
    ├── Customer_Segmentation.py
    ├── Mall_Customers.csv
    ├── Customer_Segmentation.csv
    ├── Scalar.pkl
    ├── KMeans.pkl
    └── README.md

## ⚙️ Installation and Execution

### Step 1: Install Required Libraries

```bash
pip install pandas scikit-learn matplotlib joblib