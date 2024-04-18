import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/seanohagan/projects/UACQR/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/seanohagan/projects/conformal_tree/'))

from datasets import GetDataset
from conformal_tree.conformal_tree import CTree

import helper
from tqdm import tqdm


#datasets = ['bike', 'bio', 'community', 'concrete', 'homes', 'star', 'test']
#datasets = ['test']
datasets = ['s1', 's2', 's3', 'test', 'bike', 'bio', 'community', 'concrete', 'homes', 'star', 'test']


alpha = 0.1
STANDARDIZE_RESPONSE = True

train_size = 0.40
calibration_size = 0.40
test_size = 0.20
N_RUNS = 10  # Number of runs for the simulation

# Initialize a dictionary to accumulate results
all_results = {dataset: {'isl_cp': [], 'isl_ctree': [], 'isl_studentized': []} for dataset in datasets}

for _ in tqdm(range(N_RUNS)):
    for dataset in datasets:
        X, y = GetDataset(dataset)
        if _ == 0:
            print(X.shape)
        domain = np.column_stack((np.min(X, axis=0), np.max(X, axis=0)))

        X_train, _X_temp, y_train, _y_temp = train_test_split(X, y, train_size=train_size)
        X_calib, X_test, y_calib, y_test = train_test_split(_X_temp, _y_temp, test_size=test_size / (calibration_size + test_size))

        if STANDARDIZE_RESPONSE:
            y_train /= np.mean(np.abs(y_train))
            y_calib /= np.mean(np.abs(y_calib))
            y_test /= np.mean(np.abs(y_test))

        model = RandomForestRegressor(min_samples_leaf=2, n_estimators=100, max_features='sqrt')
        model.fit(X_train, y_train)

        y_model_calib = model.predict(X_calib)
        scores = np.abs(y_calib - y_model_calib)
        gap = np.quantile(scores, 1 - alpha)

        y_model_test = model.predict(X_test)
        y_test_lb_cp = y_model_test - gap
        y_test_ub_cp = y_model_test + gap
        isl_cp = helper.average_interval_score_loss(y_test_ub_cp, y_test_lb_cp, y_test, alpha)

        ct = CTree(model, domain)
        y_lb_calib, y_ub_calib, c_g = ct.calibrate(X_calib, y_calib, y_model_calib, alpha)
        y_test_lb_ctree, y_test_ub_ctree = ct.test_interval(X_test)
        isl_ctree = helper.average_interval_score_loss(y_test_ub_ctree, y_test_lb_ctree, y_test, alpha)

        # Studentized
        spread_model = RandomForestRegressor(min_samples_leaf=2, n_estimators=100, max_features='sqrt')
        resid_response_train = np.abs(y_train - model.predict(X_train))
        spread_model.fit(X_train, resid_response_train)

        est_spreads_calib = spread_model.predict(X_calib)
        est_resids_calib = np.abs(y_calib - model.predict(X_calib))/est_spreads_calib
        qhat = np.sort(est_resids_calib)[int(np.ceil((1-alpha)*(X_calib.shape[0] + 1)))]

        y_spread_model_test = spread_model.predict(X_test)
        y_test_lb_studentized = y_model_test - (y_spread_model_test*qhat)
        y_test_ub_studentized = y_model_test + (y_spread_model_test*qhat)
        isl_studentized = helper.average_interval_score_loss(y_test_ub_studentized, y_test_lb_studentized, y_test, alpha)

        all_results[dataset]['isl_cp'].append(isl_cp)
        all_results[dataset]['isl_ctree'].append(isl_ctree)
        all_results[dataset]['isl_studentized'].append(isl_studentized)

# Calculate average results over all runs
averaged_results = []
for dataset, metrics in all_results.items():
    avg_isl_cp = np.mean(metrics['isl_cp'])
    avg_isl_ctree = np.mean(metrics['isl_ctree'])
    avg_isl_studentized = np.mean(metrics['isl_studentized'])
    averaged_results.append([dataset, avg_isl_cp, avg_isl_ctree, avg_isl_studentized])

# Create a DataFrame with the averaged results
df_results = pd.DataFrame(averaged_results, columns=['dataset', 'isl_cp', 'isl_ctree', 'isl_studentized'])

print(df_results)
print(df_results.to_latex())
