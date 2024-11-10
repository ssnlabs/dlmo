import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r"heart_failure_clinical_records_dataset.csv")

df.info()
df.head()
df.isnull().sum()

numerical_cols = df.select_dtypes(include="number").columns

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 5, i)
    sns.boxplot(df[col])
plt.tight_layout()
plt.show()

col_outliers = ["creatinine_phosphokinase", "platelets", "serum_creatinine"]

from sklearn.preprocessing import LabelEncoder
for col in col_outliers:
	if df[col].dtype=='object':
		df[col]=LabelEncoder().fit_transform(df[col])
df_encoded=df.copy()

numerical_cols = df.select_dtypes(include="number").columns

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(6, 3, i)
    sns.boxplot(df[col])
plt.tight_layout()
plt.show()

correlation_matrix = df.corr()
plt.figure(figsize = (10, 8))
sns.heatmap(correlation_matrix, annot = True)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

features = ["serum_creatinine", "time"]
x = df[features]
y = df["DEATH_EVENT"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(x_train, y_train)

svm_poly = SVC(kernel='poly', degree=3, C=1)
svm_poly.fit(x_train, y_train)

svm_rbf = SVC(kernel='rbf', gamma='auto', C=1)
svm_rbf.fit(x_train, y_train)

!pip install mlxtend

from mlxtend.plotting import plot_decision_regions
plt.figure(figsize=(18, 5))

y_train_np = y_train.to_numpy()

plt.subplot(1, 3, 1)
plot_decision_regions(x_train, y_train_np, clf=svm_linear, legend=2)
plt.title("SVM with Linear Kernel")

plt.subplot(1, 3, 2)
plot_decision_regions(x_train, y_train_np, clf=svm_poly, legend=2)
plt.title("SVM with Polynomial Kernel (degree 3)")

plt.subplot(1, 3, 3)
plot_decision_regions(x_train, y_train_np, clf=svm_rbf, legend=2)
plt.title("SVM with RBF Kernel")

plt.show()



