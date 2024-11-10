import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r'car_evaluation.csv')

df.info()
df.head()
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
for col in df.columns:
	if df[col].dtype=='object':
		df[col]=LabelEncoder().fit_transform(df[col])
df_encoded=df.copy()

correlation_matrix = df_encoded.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features = ['2.1', 'vhigh', 'low', 'vhigh.1']
x = df_encoded[features]
y = df_encoded['unacc']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)