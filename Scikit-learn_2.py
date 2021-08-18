from sklearn.datasets import load_wine
data = load_wine()
data_keys = data.keys()
print(data_keys)
data.data
print(data.DESCR)
data.feature_names
print(set(data.target))
print(len(set(data.target)))
data.target_names
import pandas as pd
X = pd.DataFrame(data.data, columns=data.feature_names)
X.head()
X.shape
X.info()
X['target'] = data.target
X.head()
X_corr = X.corr()
X_corr
high_corr = X_corr.loc[(abs(X_corr['target']) > 0.5) & (X_corr.index != 'target'), X_corr.columns != 'target'].index
high_corr
X = X.drop('target', axis=1)
X.head()
for feature_name in high_corr:
    X[f'{feature_name}_2'] = X.apply(lambda row: row[feature_name] ** 2, axis=1)
X.describe()
