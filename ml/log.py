import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r'titanic.csv')
df.shape
df.info()
df.head()
df.isnull().sum()

df = df.drop(columns=["Age","Embarked", "Cabin"])
df.shape

numerical_cols = df.select_dtypes(include="number").columns

for i, col in enumerate(numerical_cols, 1):
  plt.subplot(6, 3, i)
  sns.boxplot(df[col])
plt.tight_layout()
plt.show()

for col in ["Fare", "Parch", "SibSp"]:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        lb=q1-1.5*iqr
        ub=q3+1.5*iqr
        df=df[(df[col]>=lb)&(df[col]<=ub)]

for i, col in enumerate(numerical_cols, 1):
  plt.subplot(6, 3, i)
  sns.boxplot(df[col])
plt.tight_layout()
plt.show()

df = df.drop(columns="Parch")

from sklearn.preprocessing import LabelEncoder
for col in df.columns:
	if df[col].dtype=='object':
		df[col]=LabelEncoder().fit_transform(df[col])
df_encoded=df.copy()

correlation_matrix = df_encoded.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score

features = ["Pclass", "Sex", "Fare"]
x = df_encoded[features]
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, y_predict))
print("Precision: ", precision_score(y_test, y_predict))
print("F1-score: ", f1_score(y_test, y_predict))


