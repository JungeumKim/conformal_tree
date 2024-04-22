import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/seanohagan/projects/UACQR/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/seanohagan/projects/conformal_tree/'))

from datasets import GetDataset
from conformal_tree.conformal_tree import CTree, ConformalForest

import helper
from tqdm import tqdm


#datasets = ['bike', 'bio', 'community', 'concrete', 'homes', 'star', 'test']
#datasets = ['test']
datasets = ['s1', 's2', 's3', 'test', 'bike', 'bio', 'community', 'concrete', 'homes', 'star']


alpha = 0.1
STANDARDIZE_RESPONSE = True

train_size = 0.40
calibration_size = 0.40
test_size = 0.20
N_RUNS = 10  # Number of runs for the simulation


methods = ["cp", "ctree", "forest", "student(tree)", "student(forest)"]
metrics = ["width", "coverage", "isl"]

res = np.zeros((N_RUNS, len(datasets), len(methods), len(metrics)))

for run in tqdm(range(N_RUNS)):
    for d_idx, dataset in enumerate(datasets):
        X, y = GetDataset(dataset)
        if run == 0:
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
        n = X_train.shape[0]
        gap = np.quantile(scores, np.ceil((1 - alpha)*(n+1))/n)

        y_model_test = model.predict(X_test)
        y_test_lb_cp = y_model_test - gap
        y_test_ub_cp = y_model_test + gap

        # Conformal prediction results
        res[run, d_idx, 0, :] = [
            helper.average_width(y_test_ub_cp, y_test_lb_cp, y_test, alpha),
            helper.average_coverage(y_test_ub_cp, y_test_lb_cp, y_test),
            helper.average_interval_score_loss(y_test_ub_cp, y_test_lb_cp, y_test, alpha)
        ]

        ct = CTree(model, domain)
        y_lb_calib, y_ub_calib, c_g = ct.calibrate(X_calib, y_calib, y_model_calib, alpha)
        y_test_lb_ctree, y_test_ub_ctree = ct.test_interval(X_test)

        # Conformal Tree results
        res[run, d_idx, 1, :] = [
            helper.average_width(y_test_ub_ctree, y_test_lb_ctree, y_test, alpha),
            helper.average_coverage(y_test_ub_ctree, y_test_lb_ctree, y_test),
            helper.average_interval_score_loss(y_test_ub_ctree, y_test_lb_ctree, y_test, alpha)
        ]
        # Conformal Forest
        cf = ConformalForest(model, domain)
        cf.calibrate(X_calib, y_calib, y_model_calib, alpha)
        y_test_lb_cforest, y_test_ub_cforest = cf.combined_test_interval(X_test)

        # Conformal Forest results
        res[run, d_idx, 2, :] = [
            helper.average_width(y_test_ub_cforest, y_test_lb_cforest, y_test, alpha),
            helper.average_coverage(y_test_ub_cforest, y_test_lb_cforest, y_test),
            helper.average_interval_score_loss(y_test_ub_cforest, y_test_lb_cforest, y_test, alpha)
        ]

        # Studentized residual results
        from sklearn.tree import DecisionTreeRegressor
        spread_model = DecisionTreeRegressor(max_depth=20, max_leaf_nodes=4**(4+1), min_samples_leaf=20)

        resid_response_train = np.abs(y_train - model.predict(X_train))
        spread_model.fit(X_train, resid_response_train)

        est_spreads_calib = spread_model.predict(X_calib)
        est_resids_calib = np.abs(y_calib - model.predict(X_calib))/est_spreads_calib
        qhat = np.sort(est_resids_calib)[int(np.ceil((1-alpha)*(X_calib.shape[0] + 1)))]

        y_spread_model_test = spread_model.predict(X_test)
        y_test_lb_studentized = y_model_test - (y_spread_model_test*qhat)
        y_test_ub_studentized = y_model_test + (y_spread_model_test*qhat)

        # Studentized residual results
        res[run, d_idx, 3, :] = [
            helper.average_width(y_test_ub_studentized, y_test_lb_studentized, y_test, alpha),
            helper.average_coverage(y_test_ub_studentized, y_test_lb_studentized, y_test),
            helper.average_interval_score_loss(y_test_ub_studentized, y_test_lb_studentized, y_test, alpha)
        ]

        # Studentized residuals (forest)
        spread_model = RandomForestRegressor(min_samples_leaf=2, n_estimators=100, max_features='sqrt')

        resid_response_train = np.abs(y_train - model.predict(X_train))
        spread_model.fit(X_train, resid_response_train)

        est_spreads_calib = spread_model.predict(X_calib)
        est_resids_calib = np.abs(y_calib - model.predict(X_calib))/est_spreads_calib
        qhat = np.sort(est_resids_calib)[int(np.ceil((1-alpha)*(X_calib.shape[0] + 1)))]

        y_spread_model_test = spread_model.predict(X_test)
        y_test_lb_studentized = y_model_test - (y_spread_model_test*qhat)
        y_test_ub_studentized = y_model_test + (y_spread_model_test*qhat)

        # Studentized residual results
        res[run, d_idx, 4, :] = [
            helper.average_width(y_test_ub_studentized, y_test_lb_studentized, y_test, alpha),
            helper.average_coverage(y_test_ub_studentized, y_test_lb_studentized, y_test),
            helper.average_interval_score_loss(y_test_ub_studentized, y_test_lb_studentized, y_test, alpha)
        ]


mean_res = np.mean(res, axis=0)

metric_dataframes = {}
for metric_index, metric in enumerate(metrics):
    # Extract the data for the current metric across all methods and datasets
    data = mean_res[:, :, metric_index]

    # Create a DataFrame for the current metric
    df_metric = pd.DataFrame(data=data, columns=methods, index=datasets)
    df_metric.reset_index(inplace=True)
    df_metric.rename(columns={'index': 'dataset'}, inplace=True)

    # Store the DataFrame in the dictionary
    metric_dataframes[metric] = df_metric
    df_metric.to_csv(f"results/{metric}_results.csv", index=False)


# Print the DataFrames and convert them to LaTeX if needed
for metric, df in metric_dataframes.items():
    print(f"DataFrame for {metric}:")
    print(df)
    print(df.to_latex(index=False))
