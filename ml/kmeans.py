import pandas as pd

df = pd.read_csv('IRIS.csv')
print(df.head())

X = df.drop(columns=['species']) 

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X)

df.head()
    
print("K-Means Cluster Centers:\n", kmeans.cluster_centers_)
plt.scatter(df['sepal_length'],df['sepal_width'], c=df['kmeans_cluster'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()