# kmeans_college.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Load the dataset
# -----------------------------
df = pd.read_csv("College_Data.csv", index_col=0)

# Display first few rows
print("Initial Data Sample:")
print(df.head())

# -----------------------------
# Preprocessing
# -----------------------------

# Convert 'Private' column to numerical values: Yes -> 1, No -> 0
df['Private'] = df['Private'].map({'Yes': 1, 'No': 0})

# Store the true labels (for evaluation only, not used in training)
true_labels = df['Private']

# Remove the label column for unsupervised training
X = df.drop('Private', axis=1)

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# K-Means Clustering
# -----------------------------

# Create the KMeans model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)

# Predict the cluster for each data point
predicted_clusters = kmeans.labels_

# -----------------------------
# Evaluation
# -----------------------------

# Evaluate how well the clusters match the actual labels
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_clusters))

print("\nClassification Report:")
print(classification_report(true_labels, predicted_clusters))

# -----------------------------
# Visualization
# -----------------------------

# Visualize using two features only for simplicity
sns.scatterplot(data=df, x='Outstate', y='Room.Board', hue=predicted_clusters, palette='cool')
plt.title("K-Means Clustering of Colleges")
plt.xlabel("Out-of-State Tuition")
plt.ylabel("Room and Board Cost")
plt.legend(title='Cluster')
plt.show()
