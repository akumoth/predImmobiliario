import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_dask = dd.read_parquet('../data/processed/train')
test_dask = dd.read_parquet('../data/processed/test')
train_dataset = train_dask.compute()
test_dataset = test_dask.compute()
train_dataset = train_dataset.dropna()
X = train_dataset.drop(['state','region','price','type','lat','long'], axis=1)
y = train_dataset['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

test_dataset = test_dataset.drop(['state','region','type','lat','long'], axis=1)
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred = rfc.predict(test_dataset)

y_pred_df = pd.DataFrame(y_pred,columns=['pred'])
y_pred_df.to_csv('../data/predictions/akumoth.csv',index=False)