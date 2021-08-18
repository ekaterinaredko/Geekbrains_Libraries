#1
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
data = boston["data"]
feature_names = boston["feature_names"]
X = pd.DataFrame(data, columns=feature_names)
X.head()
target = boston["target"]
Y = pd.DataFrame(target, columns=["price"])
Y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)
check_test_lr = pd.DataFrame({"Y_test": Y_test["price"], "Y_pred_lr": y_pred_lr.flatten()})
check_test_lr.head()
from sklearn.metrics import mean_squared_error
mean_squared_error_lr = mean_squared_error(check_test_lr["Y_pred_lr"], check_test_lr["Y_test"])
print(mean_squared_error_lr)
#2
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
clf.fit(X_train, Y_train.values[:, 0])
y_pred_clf = clf.predict(X_test)
check_test_clf = pd.DataFrame({"Y_test": Y_test["price"],"Y_pred_clf": y_pred_clf.flatten()})
check_test_clf.head()
mean_squared_error_clf = mean_squared_error(check_test_clf["Y_pred_clf"], check_test_clf["Y_test"])
print(mean_squared_error_clf)
print(mean_squared_error_lr, mean_squared_error_clf)
#3
print(clf.feature_importances_)
feature_importance = pd.DataFrame({'name':X.columns, 'feature_importance':clf.feature_importances_}, columns=['feature_importance', 'name'])
feature_importance
feature_importance.nlargest(2, 'feature_importance')
#4
df = pd.read_csv('C:/Users/redkoer/Documents/Python Scripts/creditcard.csv')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
df['Class'].value_counts(normalize=True)
df.info()
pd.options.display.max_columns=100
df.head(10)
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
print('X_train ', X_train.shape)
print('X_test ', X_test.shape)
print('y_train ', y_train.shape)
print('y_test ', y_test.shape)
parameters = [{
    'n_estimators': [10, 15],
    'max_features': np.arange(3, 5),
    'max_depth': np.arange(4, 7)
}]
clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=parameters,
    scoring='roc_auc',
    cv=3,
)
clf.fit(X_train, y_train)
clf.best_params_
clf = RandomForestClassifier(max_depth=6, max_features=3, n_estimators=15)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
y_pred_proba = y_pred[:, 1]
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba)