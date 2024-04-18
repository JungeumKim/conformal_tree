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


# Assuming the helper and CTree classes are imported as follows:
# from your_module import helper, CTree

datasets = ['bike', 'bio','community', 'concrete','homes', 'star']  # No cbc, forest
alpha = 0.1  # Confidence level, adjust as needed
STANDARDIZE_RESPONSE = True


train_size = 0.40
calibration_size = 0.40
test_size = 0.20

results = []

for i,dataset in enumerate(tqdm(datasets)):
    # print(dataset)
    X, y = GetDataset(dataset)

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
    
    # Calculate the conformal prediction interval
    scores = np.abs(y_calib - y_model_calib)
    gap = np.quantile(scores, 1 - alpha)

    y_model_test = model.predict(X_test)

    y_test_lb_cp = y_model_test - gap
    y_test_ub_cp = y_model_test + gap
    isl_cp = helper.average_interval_score_loss(y_test_ub_cp, y_test_lb_cp, y_test, alpha)
    
    # Calibration using CTree
    ct = CTree(model, domain)
    y_lb_calib, y_ub_calib, c_g = ct.calibrate(X_calib, y_calib, y_model_calib, alpha)
    y_test_lb_ctree, y_test_ub_ctree = ct.test_interval(X_test)
    isl_ctree = helper.average_interval_score_loss(y_test_ub_ctree, y_test_lb_ctree, y_test, alpha)
    
    # Append results to the list
    results.append([dataset, isl_cp, isl_ctree])

# Create a DataFrame with the results
df_results = pd.DataFrame(results, columns=['dataset', 'isl_cp', 'isl_ctree'])

print(df_results)

