from joltml import Experiment, Xgboost, Sklearn
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
# Logging is done implicitly
dataset = fetch_california_housing(as_frame=True)["frame"]
dataset = dataset.dropna()
training_set = dataset[:16000]
prediction_set = dataset.iloc[16000:,:-1]

experiment = Experiment(training_set)
y = experiment.add_models([Xgboost()]).regression(target_names=['MedHouseVal'], splits=[0.8,0.2]).predict(prediction_set)
