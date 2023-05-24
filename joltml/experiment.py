import os
import json
import uuid
from datetime import datetime
from joltml.fit import Fit


class Experiment:
    '''
    from sklearn.datasets import fetch_california_housing
    # Logging is done implicitly
    dataset = fetch_california_housing(as_frame=True)["frame"]
    dataset = dataset.dropna()
    dataset = tf.random.shuffle(dataset)
    training_set = dataset[:16000]
    prediction_set = dataset[16000:,:-1]

    experiment = Experiment(training_set)
    experiment.add_models([Xgboost()])
    experiment.regression_on('MedHouseVal')
    experiment.fit(splits=[0.8,0.2])
    # Write results and default evaluation metrics
    y = experiment.predict(prediction_set)

    # Of you can do this:
    experiment = Experiment(training_set)
    y = experiment.add_models([Xgboost()]).regression_on('MedHouseVal').fit(splits=[0.8,0.2]).predict(prediction_set)
    '''

    def __init__(self, dataset=None,
                 path='./',
                 experiment_id=None,
                 utc_creation_time=None,
                 utc_last_update_time=None,
                 scaler=None,
                 models=[]) -> None:
        self.path = path
        if not os.path.isdir(path + 'jolt_lab'):
            os.mkdir(path + 'jolt_lab')
        if experiment_id is None:
            self.experiment_id = str(uuid.uuid4())
        else:
            self.experiment_id = experiment_id
        self.dataset = dataset
        self.models = models
        self.utc_creation_time = str(utc_creation_time or datetime.utcnow())
        self.utc_last_update_time = utc_last_update_time

        self.lab_book = {
            'experiment_id': self.experiment_id
        }
        if not os.path.isdir(path + '/jolt_lab/' + self.experiment_id):
            os.mkdir(path + '/jolt_lab/' + self.experiment_id)
        else:
            # Read the experiment data
            pass

        lab_book_f = open(path + '/jolt_lab/' + 
                          self.experiment_id + '/lab_book.json', 'w')
        json.dump(self.lab_book, lab_book_f)
        lab_book_f.close()

    def load_experiment(self, experiment_id):
        pass

    def add_data(self, dataset):
        if not self.dataset:
            self.dataset = dataset

    def add_models(self, models):
        for model in models:
            self.models += [model]
        return self

    def _prepare_fit(self, splits, inputs, targets, scaler=None):
        return Fit(self.experiment_id, self.dataset, splits, inputs, targets, path=self.path, scaler=scaler)

    def regression(self, splits=None, inputs=None, targets=None, metrics=None, scaler=None):
        # Starts a fitting run
        fitting_run = self._prepare_fit(splits, inputs, targets, scaler)
        for i in range(len(self.models)):
            fitting_run.regression(self.models[i], metrics)
            fitting_run.save_model(self.models[i])
            fitting_run.save_metrics(self.models[i])
        return self

    def regression_optimize(self, splits=None, inputs=None, targets=None, metrics=None, scaler=None, params=None, n_trials=10, score=None):
        # Starts a fitting run
        fitting_run = self._prepare_fit(splits, inputs, targets, scaler)
        for i in range(len(self.models)):
            fitting_run.regression_optimize(
                self.models[i], metrics, params, n_trials, score)
            fitting_run.save_model(self.models[i])
            fitting_run.save_metrics(self.models[i])
        return self

    def predict(self, X):
        y = []
        for model in self.models:
            y += [model.predict(X)]
        return y
