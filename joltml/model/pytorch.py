import torch
import torch.nn as nn
from joltml.model import Model
import optuna
from joltml.joltmeter import RegressionMetrics, ClassificationMetrics
from joltml.version import VERSION
import pandas as pd
import numpy as np
import tqdm
from captum.attr import IntegratedGradients


class Pytorch(Model):
    def __init__(self, native_model=None,
                 joltml_version: str = ...,
                 model_path='.',
                 utc_time_created=None,
                 model_id=None, **kwargs):
        super().__init__(native_model, joltml_version,
                         model_path, utc_time_created, model_id, **kwargs)
        self.epochs = 100
        if 'epochs' in kwargs:
            self.epochs = kwargs['epochs']
        self.every_epochs = 10
        if 'every_epochs' in kwargs:
            self.every_epochs = kwargs['every_epochs']
        self.lr = 0.0001
        if 'lr' in kwargs:
            self.lr = kwargs['lr']
        self.batch_size = 10  # size of each batch
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
            
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.native_model.parameters(), lr=self.lr)

    def fit(self, X, y, X_test, y_test):

        if isinstance(X, pd.DataFrame):
            X = torch.from_numpy(np.array(X.values,np.float32))
        if isinstance(y, pd.DataFrame):
            y = torch.from_numpy(np.array(y.values,np.float32))
        if isinstance(X_test, pd.DataFrame):
            X_test = torch.from_numpy(np.array(X_test.values,np.float32))
        if isinstance(y_test, pd.DataFrame):
            y_test = torch.from_numpy(np.array(y_test.values,np.float32))

        batch_start = torch.arange(0, len(X), self.batch_size)
        
        self.train_loss_history = []
        self.test_loss_history = []
        for epoch in range(self.epochs):
            self.native_model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                for start in bar:
                    X_batch = X[start:start+self.batch_size]
                    y_batch = y[start:start+self.batch_size]
                    y_pred = self.native_model(X_batch)
                    loss = self.loss_function(y_pred, y_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            self.native_model.eval()
            with torch.inference_mode():
                test_pred = self.native_model(X_test)
                test_loss = self.loss_function(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
                if epoch % self.every_epochs == 0:
                    self.train_loss_history.append(loss.detach().numpy())
                    self.test_loss_history.append(test_loss.detach().numpy())
                    print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss} ")
    
    def fit_optimize(self, X, y, X_test, y_test, params, n_trials, score):
        return super().fit_optimize(X, y, X_test, y_test, params, n_trials, score)
    
    def classification_on(self):
        return super().classification_on()
    
    def regression_on(self):
        return super().regression_on()
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = torch.from_numpy(np.array(X.values,np.float32))
        return self.native_model(X)
    
    def save_model(self, path):
        return super().save_model(path)
    
    def prepare_data(self, X):
        return X.detach().numpy()
    
    def explain(self):
        pass
