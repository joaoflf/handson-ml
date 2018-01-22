# %%
import os
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
# from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import (GridSearchCV, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join("datasets", "titanic")
train_set = pd.read_csv(os.path.join(DATA_PATH, "titanic-training.csv"), index_col='PassengerId')
test_set = pd.read_csv(os.path.join(DATA_PATH, "titanic-test.csv"), index_col='PassengerId')


pipeline = Pipeline([
    ('one_hot', OneHotEncoder(cols=['Sex', 'Embarked']))
])

X_train = train_set.drop(["Survived", "Name", "Ticket", "Cabin"], axis=1)
X_test = test_set.drop(["Name", "Ticket", "Cabin"], axis=1)


y_train = train_set["Survived"].copy()


X_train = pipeline.fit_transform(X_train)
X_test = pipeline.fit_transform(X_test)

X_train = X_train.drop(["Embarked_3"], axis=1)


X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
X_cv = np.nan_to_num(X_cv)

#%%
param_grid = [{'n_estimators':[3, 10, 30], 'max_features':np.arange(3, 10)}]
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

y_pred1 = cross_val_predict(grid, X_cv, y_cv, cv=3)

feature_importances = grid.best_estimator_.feature_importances_
sorted(feature_importances, reverse=True)


class ImportantFeaturesFilter(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances):
        self.feature_importances = feature_importances
    def fit(self):
        return self
    def transform(self, X):
        idx = np.asarray(np.argwhere(self.feature_importances > 0.04).flatten())
        return X[:, idx]

important_features_pipeline = Pipeline([
    ('filter', ImportantFeaturesFilter(feature_importances)),
])


# X_train = important_features_pipeline.fit_transform(X_train)
# X_test = important_features_pipeline.fit_transform(X_test)

# X_cv = important_features_pipeline.fit_transform(X_cv)
# # cross_val_score(grid, X_cv, y_cv, cv=3, scoring="accuracy")
# grid.fit(X_train, y_train)
# y_pred2 = cross_val_predict(grid, X_cv, y_cv, cv=3)

# print(f1_score(y_cv, y_pred1))
# print(f1_score(y_cv, y_pred2))
prediction = grid.predict(X_test)
prediction = pd.DataFrame(prediction, columns=['predictions']).to_csv('prediction.csv')
