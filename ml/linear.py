import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r"Housing.csv")
df.info()
df.head()
df.isnull().sum()

numerical_cols = df.select_dtypes(include="number").columns

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(df[col])
plt.tight_layout()
plt.show()

for col in ['price','area']:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr=q3-q1
        lb=q1-1.5*iqr
        ub=q3+1.5*iqr
        df=df[(df[col]>=lb)&(df[col]<=ub)]


col_outliers=['price','area']
for i, col in enumerate(col_outliers, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(df[col])
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder
for col in df.columns:
	if df[col].dtype=='object':
		df[col]=LabelEncoder().fit_transform(df[col])
df_encoded=df.copy()

df_encoded.head()

correlation_matrix = df_encoded.corr()
plt.figure(figsize = (10, 8))
sns.heatmap(correlation_matrix, annot = True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score

features = ["area", "bathrooms", "stories"]
x = df[features]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("MSE: ", mean_squared_error(y_test, y_predict))
print("R score",r2_score(y_test,y_predict))

sns.regplot(x='area', y='price', data=df)


