import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r'diabetes.csv')

df.info()
df.head()
df.isnull().sum()

numerical_cols = df.select_dtypes(include="number").columns

for i, col in enumerate(numerical_cols, 1):
  plt.subplot(6, 3, i)
  sns.boxplot(df[col])
plt.tight_layout()
plt.show()

outlier_cols = ["Pregnancies", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
for col in outlier_cols:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        lb=q1-1.5*iqr
        ub=q3+1.5*iqr
        df=df[(df[col]>=lb)&(df[col]<=ub)]

numerical_cols = df.select_dtypes(include="number").columns

for i, col in enumerate(numerical_cols, 1):
  plt.subplot(6, 3, i)
  sns.boxplot(df[col])
plt.tight_layout()
plt.show()

correlation_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

features = ["Glucose", "BMI", "Pregnancies", "Age"]
x = df[features]
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

k = 6
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


