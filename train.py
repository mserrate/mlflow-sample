import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import mlflow
from mlflow import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# these are the column labels from the census data files
COLUMNS = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income-level'
)

# categorical columns contain data that need to be turned into numerical
# values before being used by XGBoost
CATEGORICAL_COLUMNS = (
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
)

# load training set
with open('./census_data/adult.data', 'r') as train_data:
    raw_training_data = pd.read_csv(train_data, header=None, names=COLUMNS)
# remove column we are trying to predict ('income-level') from features list
train_features = raw_training_data.drop('income-level', axis=1)
# create training labels list
train_labels = (raw_training_data['income-level'] == ' >50K')

# load test set
with open('./census_data/adult.test', 'r') as test_data:
    raw_testing_data = pd.read_csv(test_data, names=COLUMNS, skiprows=1)
# remove column we are trying to predict ('income-level') from features list
test_features = raw_testing_data.drop('income-level', axis=1)
# create training labels list
test_labels = (raw_testing_data['income-level'] == ' >50K.')

# convert data in categorical columns to numerical values
encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
for col in CATEGORICAL_COLUMNS:
    train_features[col] = encoders[col].fit_transform(train_features[col])
for col in CATEGORICAL_COLUMNS:
    test_features[col] = encoders[col].fit_transform(test_features[col])

# that's a trick to serve the model without ordering problems with params by json
train_features = train_features.reindex(sorted(train_features.columns), axis=1)
test_features = test_features.reindex(sorted(test_features.columns), axis=1)

with mlflow.start_run():
    # model parameters
    params = {'learning_rate': 0.1, 'n_estimators': 100, 'seed': 0, 'subsample': 1, 'colsample_bytree': 1,
                  'objective': 'binary:logistic', 'max_depth': 3}

    # log model params
    for key in params:
        mlflow.log_param(key, params[key])

    # train XGBoost model
    gbtree = XGBClassifier(**params)
    gbtree.fit(train_features, train_labels)

    importances = gbtree.get_booster().get_fscore()
    print(importances)

    # get predictions
    y_pred = gbtree.predict(test_features)

    accuracy = accuracy_score(test_labels, y_pred)
    print("Accuracy: %.1f%%" % (accuracy * 100.0))

    # log accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    sns.set(font_scale=1.5)
    xgb.plot_importance(gbtree)
    plt.savefig("importance.png", dpi = 200, bbox_inches = "tight")

    mlflow.log_artifact("importance.png")

    # log model
    mlflow.sklearn.log_model(gbtree, "model")