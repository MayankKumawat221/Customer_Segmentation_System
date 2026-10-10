import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import joblib


# Load the data set

data = pd.read_csv('Mall_Customers.csv')
print('Top 5 rows of csv file:\n',data.head())
print('Shape of data set: ',data.shape)


# Data Understanding

print('Type of all columns:',data.columns)
print('Total missing value:',data.isna())


# Data Cleaning

print('Total Duplicate row in data set:',data.duplicated().sum())
print('Drop Duplicate row in data set if present:',data.drop_duplicates())


# Exploratory Data Analysis

print('Statestical Data Distribution:',data.describe())


# Feature Selection

X = data.drop(columns='CustomerID')
print('Feature Dimension:',X.shape)
print(X.groupby('Gender')['Annual Income'].agg(count = 'count',mean = 'mean'))
print(X.groupby('Gender')['Spending Score'].agg(count = 'count',mean = 'mean'))
# Through Average Annual Income and Average Spending Score we conclude that difference between male and female is very small so we drop Gender column
X = X.drop(columns='Gender')
print(X.shape)


# Feature Scaling

Scalar = StandardScaler()
X_Scale = Scalar.fit_transform(X)
print(' X after Scaling:\n',X_Scale)
# for k in [0,1,2]:
#     print('Mean of X_Scale:',k,':',X_Scale[:,k].mean())
#     print('Standard Deviation of X_Scale:',k,':',X_Scale[:,k].std())


# Determine the Optimal Number of Clusters

# inertia_values = []
# for k in range(3,11):
#     model = KMeans(random_state=42,n_clusters=k,n_init=10)  
#     model.fit(X_Scale)
#     inertia_values.append(model.inertia_)
# plt.plot(range(3,11),inertia_values,marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.show()
# from above Elbow analysis we conclude that 6 cluster are good for model.
model = KMeans(random_state=42,n_clusters=6,n_init=5)  
model.fit(X_Scale)


# Assign Customers to Clusters

print(model.labels_)
data['labels'] = model.labels_
print(data)


# Visualization Customer Clusters

plt.scatter(data['Annual Income'],data['Spending Score'],c=data['labels'])
plt.xlabel('Annual Income of Customer')
plt.ylabel('Spending Score')
plt.title('Visualization of Customer Cluster')
plt.show()


# Interpret Customer Segments

customer_segments = data.groupby('labels').agg({ 'Age' : 'mean','Annual Income' : 'mean','Spending Score' : 'mean'})
print('Compare the mean of all labels:',customer_segments)
cluster_name = {
    0 : 'Older_Medium_Income_Medium_Spending',
    1 : 'Young_Medium_Income_Medium_Spending',
    2 : 'Middle_Medium_Income_Medium_Spending',
    3 : 'Young_Medium_Income_Medium_Spending',
    4 : 'Young_Medium_Income_Medium_Spending',
    5 : 'Middle_Medium_Income_Medium_Spending'
}
data['Customer_Segments'] = data['labels'].map(cluster_name)
print(data.head(10))


# Evaluate Clustering Results

Score = silhouette_score(X_Scale,model.labels_)
print('Silhouette_score',Score)
# silhouette score ka result hai 0.42 mtlb thoda improvement required hai.


# Prepare the Final Customer Segmentation Output

print('Top 10 rows of dataset:\n',data.head(10))
print('Labels and Customer_Segments\n',data[['labels','Customer_Segments']])
print('Checking missing value:',data.isna().sum())
print('Displaying all dataset:\n',data)


# Save the Final Customer Segmentation Results

data.to_csv('Customer_Segmentation.csv',index=False)
check = pd.read_csv('Customer_Segmentation.csv')
print(data.columns==check.columns)
joblib.dump(Scalar,'Scalar.pkl')
joblib.dump(model,'KMeans.pkl')



# Test the Segmentation on New Customer Data

new_data = pd.DataFrame({
    'Age':[25,55,20],
    'Annual Income':[35,30,26],
    'Spending Score':[45,15,50]
})
new_data_scale = Scalar.transform(new_data)
new_labels = model.predict(new_data_scale)
print(new_labels.shape)
new_data['labels'] = new_labels
new_data['Customer_Segmentation'] = new_data['labels'].map(cluster_name)
print(new_data)