import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Dataset : https://www.kaggle.com/datasets/dragonheir/logistic-regression?resource=download
data = pd.read_csv('data.csv')
refined = data.drop('User ID',axis=1)
# print(refined.head())
# plt.scatter([data.iloc[:,3]],data.iloc[:,4])
# plt.show()
vals = data[['Gender','Age','EstimatedSalary']]
target = data['Purchased']
# print(target.head())
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(drop='first'),['Gender'])],remainder='passthrough')
# print(X_encoded)
pipeline = Pipeline([('ct',ct),('lgt',LogisticRegression())])
# X_encoded = ct.fit_transform(vals)
X_train , X_test , Y_train , Y_test = train_test_split(vals,target,train_size=0.7,random_state=42) # We use random state to stop the score from changing when we use random state
pipeline.fit(X_train,Y_train)
final = pipeline.predict(X_test)
print(pipeline.score(X_test,Y_test))
