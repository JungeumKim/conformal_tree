import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor, KNeighborsQuantileRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.stats import norm, uniform, beta, iqr, mode
import math
from functools import partial
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import QuantileRegressor, LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
from scipy.special import expit, logit
from lightgbm import LGBMRegressor
from scipy.spatial.distance import cdist
from oqrhelper import HSIC, wsc_unbiased


def generate_data(n, p, cond_exp, noise_sd_fn, x_dist = partial(np.random.uniform, low=0, high=10)):
    x = x_dist(size=n*p).reshape(n,p)
    noise_sd = noise_sd_fn(x)
    noise = np.random.normal(scale=noise_sd, size=n)
    y = cond_exp(x)+noise
    return x,y

def interval_score_loss(high_est, low_est, actual, alpha):
    return high_est - low_est + 2/alpha*(low_est - actual)*(actual < low_est)+2/alpha*(actual - high_est)*(actual > high_est)

def average_interval_score_loss(high_est, low_est, actual, alpha):
    return np.mean(interval_score_loss(high_est, low_est, actual, alpha))

def coverage_indicators(high_est, low_est, actual):
    return (high_est>=actual)&(low_est<=actual)

def average_coverage(high_est, low_est, actual):
    return np.mean(coverage_indicators(high_est,low_est, actual))

def interval_widths(high_est, low_est):
    return high_est - low_est

def average_interval_width(high_est, low_est):
    return np.mean(interval_widths(high_est, low_est))

def corr_coverage_widths(high_est, low_est, actual):
    coverage_indicator_vector = coverage_indicators(high_est, low_est, actual)
    widths_vector = interval_widths(high_est, low_est)
    return np.abs(np.corrcoef(coverage_indicator_vector, widths_vector)[0,1])

def hsic_coverage_widths(high_est, low_est, actual):
    coverage_indicator_vector = coverage_indicators(high_est, low_est, actual)
    if isinstance(coverage_indicator_vector, pd.Series):
        coverage_indicator_vector = coverage_indicator_vector.to_numpy()

    coverage_indicator_vector = torch.tensor(coverage_indicator_vector, dtype=torch.float).unsqueeze(1)
    widths_vector = torch.tensor(interval_widths(high_est, low_est)).unsqueeze(1)
    return np.abs(HSIC(coverage_indicator_vector, widths_vector).numpy())

def wsc(X, y, high_est, low_est):
    coverage = average_coverage(high_est, low_est, y)

    return np.abs(wsc_unbiased(X, y, high_est, low_est) - coverage)

def randomized_conformal_cutoffs(series, T, alpha=0.1):
    series = series.sort_values(ascending=True)


    def open_interval(num):
        return num - 1e-06
        # return np.nextafter(num, -np.inf)

    n1 = series.shape[0]

    k = math.ceil((1-alpha)*(n1+1))

    delta = k - (1-alpha)*(n1+1)

    tau = np.random.uniform(size=T)

    threshold = np.empty_like(tau)

    T0 = (series.iloc[:k-1]==series.iloc[k-1-1]).sum()
    T1 = (series.iloc[k-1:]==series.iloc[k-1]).sum()

    if series.iloc[k-1-1]<series.iloc[k-1]:
        cutoff_1 = delta/(T0+1)
        cutoff_2 = delta*T0/(T0+1)
        cutoff_3 = (1-delta)*T1/(1+T1)
        condition_1 = tau<cutoff_1
        condition_2 = np.logical_and(tau>=cutoff_1, tau < cutoff_1+cutoff_2)
        condition_3 = np.logical_and(tau>= cutoff_1+cutoff_2, tau < cutoff_1+cutoff_2+cutoff_3)
        condition_4 = tau>=cutoff_1+cutoff_2+cutoff_3
        # np.nextafter here will find largest float strictly less than input
        threshold[condition_1] = open_interval(series.iloc[k-1-1])
        threshold[condition_2] = series.iloc[k-1-1]
        threshold[condition_3] = open_interval(series.iloc[k-1])
        threshold[condition_4] = series.iloc[k-1]
    else:
        cutoff_1 = (delta+T1)/(1+T0+T1)
        condition_1 = tau<cutoff_1
        threshold[condition_1] = open_interval(series.iloc[k-1])
        threshold[np.logical_not(condition_1)] = series.iloc[k-1]

    return threshold

def select_column_per_row(arr, cols):
    # turn cols into np array if not, preserve shape if already
    cols = np.array([cols]).reshape(-1)

    if arr.shape[0] % cols.shape[0] != 0:
        raise Exception("need one column specified per row, or just one column")
    row_idxs = np.arange(arr.shape[0])[:,None]

    return arr[row_idxs, cols[:,None]].reshape(-1)




class sample_binning_model():
    def __init__(self, q_lower=5, q_upper=95):
        self.q_lower = q_lower
        self.q_upper = q_upper

    def fit(self, x_train, y_train):
        self.x_train_values = np.unique(x_train, axis=0)

        self.bin_model_lower = np.empty_like(self.x_train_values).astype(float)
        self.bin_model_upper = np.empty_like(self.x_train_values).astype(float)
        for t,x in enumerate(self.x_train_values):
            self.bin_model_lower[t] = np.quantile(y_train[(x_train==x).reshape(-1)], self.q_lower/100)
            self.bin_model_upper[t] = np.quantile(y_train[(x_train==x).reshape(-1)], self.q_upper/100)

    def predict(self, x_calib):
        y_upper_calib = np.empty(x_calib.shape[0])
        y_lower_calib = np.empty(x_calib.shape[0])

        for t,x in enumerate(self.x_train_values):
            y_lower_calib[(x_calib==x).reshape(-1)] = self.bin_model_lower[t]
            y_upper_calib[(x_calib==x).reshape(-1)] = self.bin_model_upper[t]

        return [y_lower_calib, y_upper_calib]

class QuantileBandwidthModel:
    def __init__(self, quantiles, threshold):
        self.quantiles= quantiles
        self.threshold = threshold
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("You must fit the model before making predictions.")

        y_pred = []
        for x_test in X_test:
            distances = cdist([x_test], self.X_train, metric='euclidean')
            mask = distances[0] <= self.threshold
            selected_y_train = self.y_train[mask]

            if len(selected_y_train) == 0:
                raise ValueError(f"No training samples found within the threshold of {self.threshold} for test point {x_test}.")

            y_pred.append(np.quantile(selected_y_train, self.quantiles))

        y_pred = np.array(y_pred)
        return [y_pred[:,c] for c in range(y_pred.shape[1])]

class CatBoostWrapper:
    def __init__(self, model, t=None):
        self.lower_model = model[0]
        self.upper_model = model[1]
        if t is None:
            self.t = model[0].tree_count_ - 1
        else:
            self.t = t

    def predict(self, X_test):
        return self.lower_model.predict(X_test, ntree_end=self.t+1), self.upper_model.predict(X_test, ntree_end=self.t+1)



class QuantileRegressionNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100, dropout=False, batch_norm=False):
        super(QuantileRegressionNet, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        if dropout:
            self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        if batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        if dropout:
            self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout:
            x = self.dropout2(x)
        x = self.fc3(x)
        return x

class QuantileRegressionNN:
    def __init__(self, quantiles=[0.5], lr=1e-3, epochs=100, batch_size=32, dropout=0, normalize=True,
                 weight_decay=0, hidden_size=100, batch_norm=True, gamma=0.999, step_size=10,random_state=None,
                 epoch_model_tracking=False, verbose=False, use_gpu=True, undo_quantile_crossing=False,
                 drop_last=False, running_batch_norm=False, train_first_batch_norm=False):
        self.quantiles = quantiles
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.normalize = normalize
        self.dropout=dropout
        self.weight_decay=weight_decay
        self.hidden_size=hidden_size
        self.random_state = random_state
        self.batch_norm = batch_norm
        self.gamma = gamma
        self.step_size = step_size
        self.epoch_model_tracking = epoch_model_tracking
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.undo_quantile_crossing = undo_quantile_crossing
        self.drop_last = drop_last
        self.running_batch_norm = running_batch_norm
        self.train_first_batch_norm = train_first_batch_norm
        if random_state:
            torch.manual_seed(random_state)

    def fit(self, X, y, X_val = None, y_val = None):
        X = np.array(X)
        y = np.array(y)
        if self.normalize:
            self.x_min = X.min(axis=0)
            self.x_max = X.max(axis=0)
            self.x_range = self.x_max - self.x_min
            self.x_range[self.x_range==0] = 1
            self.y_min = y.min()
            self.y_max = y.max()
            X = (X - self.x_min) / self.x_range
            y = (y - self.y_min) / (self.y_max - self.y_min)

            if y_val is not None:
                y_val = (y_val - self.y_min) / (self.y_max - self.y_min)

        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.net = QuantileRegressionNet(input_size=X.shape[1], output_size=len(self.quantiles),
                                        dropout=self.dropout, hidden_size=self.hidden_size,
                                        batch_norm=self.batch_norm)

        self.net.to(self.device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                             drop_last=self.drop_last)

        y_pred_val_across_epochs = []
        self.saved_models = []
        for epoch in range(self.epochs):
            epoch_losses=[]
            if self.running_batch_norm and not(self.train_first_batch_norm):
                for m in self.net.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()
            elif self.running_batch_norm and epoch>0:
                for m in self.net.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()

            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self.net(X_batch)
                loss = 0.0
                for i, q in enumerate(self.quantiles):
                    error = y_batch - y_pred[:, i]
                    if q == 'mean':
                        loss += torch.square(error).mean()
                    else:
                        loss += torch.max((q - 1) * error, q * error).mean()
                with torch.no_grad():
                    epoch_losses.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()

            if X_val is not None and y_val is not None:
                preds = self.predict(X_val, use_seed=False, undo_normalization=False)
                loss_val=0.0
                for i, q in enumerate(self.quantiles):
                    error = y_val - preds[i]
                    if q == 'mean':
                        loss_val += np.square(error).mean()
                    else:
                        loss_val += np.maximum((q - 1) * error, q * error).mean()
                if self.verbose:
                    print(f"Epoch: {epoch} \t Train Loss: {np.mean(epoch_losses)} Validation Loss: {loss_val}")
                y_pred_val_across_epochs.append(preds)

            self.scheduler.step()
            if self.epoch_model_tracking:
                self.saved_models.append(copy.deepcopy(self.net.state_dict()))

        if y_pred_val_across_epochs != []:

            self.y_pred_val_across_epochs = np.stack(y_pred_val_across_epochs)

                # self.net.train()

    def predict(self, X, ensembling=None, use_seed=True, undo_normalization=True):
        if use_seed and self.random_state:
            torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        if self.x_min is not None and self.x_max is not None:
            X = (X - self.x_min) / self.x_range
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if ensembling and not(self.epoch_model_tracking):
            self.net.train()
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

            y_pred = list()

            with torch.no_grad():
                for t in range(ensembling):
                    X_out = self.net(X)
                    y_pred.append(X_out.cpu().squeeze())

            y_pred = torch.stack(y_pred)

        elif ensembling and self.epoch_model_tracking:
            y_pred = list()
            for state_dict in self.saved_models[-ensembling:]:
                self.net.load_state_dict(state_dict)
                self.net.eval()
                with torch.no_grad():
                    X_out = self.net(X)
                    y_pred.append(X_out.cpu().squeeze())

            y_pred = torch.stack(y_pred)

        else:
            self.net.eval()

            with torch.no_grad():
                y_pred = self.net(X)

        y_pred = y_pred.detach().cpu().numpy()
        if self.y_min is not None and self.y_max is not None and undo_normalization:
            y_pred = y_pred * (self.y_max - self.y_min) + self.y_min

        if self.undo_quantile_crossing and ensembling:
            y_pred[:,:,0][y_pred[:,:,0]>y_pred[:,:,-1]] = 0.5*y_pred[:,:,-1][y_pred[:,:,0]>y_pred[:,:,-1]] + 0.5*y_pred[:,:,0][y_pred[:,:,0]>y_pred[:,:,-1]]
            y_pred[:,:,-1][y_pred[:,:,0]>y_pred[:,:,-1]] = 0.5*y_pred[:,:,-1][y_pred[:,:,0]>y_pred[:,:,-1]] + 0.5*y_pred[:,:,0][y_pred[:,:,0]>y_pred[:,:,-1]]
        elif self.undo_quantile_crossing:
            y_pred[:,0][y_pred[:,0]>y_pred[:,-1]] = 0.5*y_pred[:,-1][y_pred[:,0]>y_pred[:,-1]] + 0.5*y_pred[:,0][y_pred[:,0]>y_pred[:,-1]]
            y_pred[:,-1][y_pred[:,0]>y_pred[:,-1]] = 0.5*y_pred[:,-1][y_pred[:,0]>y_pred[:,-1]] + 0.5*y_pred[:,0][y_pred[:,0]>y_pred[:,-1]]

        self.net.train()

        if ensembling:
            return np.moveaxis(y_pred, [0,1,2], [2,1,0])
        else:
            return np.moveaxis(y_pred, [0], [1])

    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        alpha = self.quantiles[0] + 1 - self.quantiles[-1]

        preds = self.predict(X)

        return np.mean(interval_score_loss(preds[-1], preds[0], y, alpha))
