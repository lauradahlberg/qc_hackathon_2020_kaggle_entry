import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


def save_prediction_to_csv(y_pred):
    """
    Use this function to save your prediction result to a csv file.
    The resulting csv file is named as [team_name].csv

    :param y_pred: an array or a pandas series that follows the SAME index order as in the testing data
    """
    pd.DataFrame(dict(
        target=y_pred
    )).to_csv('predictions.csv', index=False, header=False)


# load data
training_data = pd.read_csv('training.csv', index_col=0)
testing_data = pd.read_csv('testing.csv', index_col=0)

# build and fit pipeline
y = training_data.target
X = training_data.drop(columns=['target'])

pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('regressor', GradientBoostingRegressor(n_estimators=10))
])
pipeline.fit(X, y)

# make prediction and save result to csv for submission
y_pred = pipeline.predict(testing_data)
save_prediction_to_csv(y_pred)
