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

from helper import generate_data, interval_score_loss, randomized_conformal_cutoffs, select_column_per_row
from uacqr import uacqr

class experiment():
    def __init__(self, fixed_model_params=None, n=None, p=None, T=None, X = None, y = None, cond_exp=None, noise_sd_fn=None, x_dist=None, 
                 S=25, random_state=42, var_name = 'min_samples_leaf', var_list = [1,5], metric='interval_score_loss',
                 fast_uacqr=True, B=100, model_type='rfqr', uacqrs_agg='std', empirical_data_fraction=0.01,
                 oracle_g=None, oracle_t=None, inject_noise=False, randomized_conformal=False, 
                 extraneous_quantiles=[0.1, 0.25, 0.4, 0.5, 0.6, 0.75,0.9], 
                 uacqrs_bagging=False, sorting_column=0, max_normalization=False,
                 local_metric=False, oqr_metrics=False, file_name=None):
        
        self.var_name = var_name
        self.metric = metric
        self.S = S
        self.fast_uacqr = fast_uacqr
        self.sorting_column = sorting_column
        self.local_metric = local_metric
        if self.metric == 'conditional_coverage':
            self.local_metric = True

        if file_name:
            self.results_df = pd.read_pickle(file_name)
            if len(self.results_df.columns) == 5:
                # self.metric = 'conditional_coverage'
                self.local_metric = True
                self.S = self.results_df.index.value_counts().mode()[0]
                return
            self.var_name = self.results_df.columns[2]
            self.S = int(self.results_df['alpha'].count() / self.results_df.nunique()[self.var_name])
            return

        col_names = ['alpha', 'cqr_score_threshold', 'uacqrs_score_threshold', 'uacqrp_test_coverage', 
                                   'cqr_test_coverage', 'uacqrs_test_coverage', 'median_test_coverage', 'uacqrp_average_length_test', 
                                   'cqr_average_length_test', 'uacqrs_average_length_test', 'cqrr_average_length_test',
                                   'median_average_length_test', 'uacqrp_test_len_std', 
                                   'cqr_test_len_std', 'median_test_len_std', 'uacqrp_interval_score_loss', 'cqr_interval_score_loss', 
                                   'uacqrs_interval_score_loss', 'cqrr_interval_score_loss','median_interval_score_loss',
                                   'base_average_length_test','base_test_coverage','cqrr_test_coverage',
                                   'base_oqr_corr', 'uacqrp_oqr_corr','uacqrs_oqr_corr','cqr_oqr_corr','cqrr_oqr_corr',
                                   'base_oqr_hsic', 'uacqrp_oqr_hsic','uacqrs_oqr_hsic','cqr_oqr_hsic','cqrr_oqr_hsic',
                                   'base_oqr_wsc', 'uacqrp_oqr_wsc','uacqrs_oqr_wsc','cqr_oqr_wsc','cqrr_oqr_wsc',var_name]
        

        df = pd.DataFrame(columns=col_names)

        t=0

        df_list = []
        for var in var_list:
            sim_model_params = fixed_model_params.copy()
            sim_model_params[var_name] = var

            if var_name == 'p':
                p = var
                del sim_model_params['p']

            for s in range(random_state, random_state+S):
                np.random.seed(s)
                
                

                q_lower = 5
                q_upper = 95
                alpha = 0.1
                
                if X is not None:
                    x_train, y_train, x_calib, y_calib, x_test, y_test = self.empirical_data_draw(X, y, empirical_data_fraction, s,
                                                                                                    max_normalization)
                    
                elif cond_exp is not None:
                    x_train, y_train, x_calib, y_calib, x_test, y_test = self.simulate_data(cond_exp, noise_sd_fn, x_dist, n, p, T)
                    self.p=p
                else:
                    raise Exception("Need to either provide existing data or specify a DGP")
                
                
                uacqr_results = uacqr(model_params=sim_model_params, 
                                bootstrapping_for_uacqrp=not(fast_uacqr), B=B, random_state=s, model_type=model_type, uacqrs_agg=uacqrs_agg,
                                oracle_g=oracle_g, randomized_conformal=randomized_conformal, extraneous_quantiles=extraneous_quantiles,
                                uacqrs_bagging=uacqrs_bagging)
                uacqr_results.fit(x_train, y_train)
                if oracle_g:
                    uacqr_results.calibrate(x_calib, y_calib, inject_noise=inject_noise, cond_exp=cond_exp, noise_sd_fn=noise_sd_fn)
                else:
                    uacqr_results.calibrate(x_calib, y_calib, inject_noise=inject_noise)
                uacqr_results.evaluate(x_test, y_test, oqr_metrics=oqr_metrics)


                if self.local_metric:
                    uacqr_results.calculate_conditional_coverage(cond_exp, noise_sd_fn, plot=False, sorting_column=sorting_column,
                                                                    metric=metric)
                    df_list.append(uacqr_results.conditional_coverage)
                
                else:
                
                    single_run_results = vars(uacqr_results)

                    df.loc[t] = {k: v for k, v in single_run_results.items() if isinstance(v,(float,np.floating))}
                    df.loc[t,var_name] = var 


                t+= 1

                if t % 5 ==0:
                    print(t)

        if self.local_metric:
            self.results_df = pd.concat(df_list)
        else:
            df = df.melt(id_vars=var_name,ignore_index=False)


            df[['method','metric']] = df.variable.str.split('_',n=1, expand=True)

            df.drop('variable',axis=1, inplace=True)

            df.reset_index(inplace=True)

            df = df.pivot(index=['index','metric', var_name],columns='method', values='value').reset_index()
            
            df.rename(columns={"cqr":"CQR","uacqrs":"UACQR-S","uacqrp":"UACQR-P", "cqrr":"CQR-r"}, inplace=True)

            self.results_df = df
            
        
        self.last_run = uacqr_results

        self.results_df['CQR'] = pd.to_numeric(self.results_df['CQR'])
        self.results_df['UACQR-S'] = pd.to_numeric(self.results_df['UACQR-S'])
        self.results_df['UACQR-P'] = pd.to_numeric(self.results_df['UACQR-P'])
        self.results_df['CQR-r'] = pd.to_numeric(self.results_df['CQR-r'])

    def empirical_data_draw(self, X, y, empirical_data_fraction, s, max_normalization=False):

        
        train_fraction = 0.4
        test_fraction = 1/3
        calib_fraction = 1


        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            y = pd.Series(y.reshape(-1))

        X_drawn = X.sample(frac=empirical_data_fraction, random_state=s)

        x_train = X_drawn.sample(frac=train_fraction, random_state=s)

        x_test = X_drawn.drop(x_train.index).sample(frac=test_fraction, random_state=s)

        x_calib = X_drawn.drop(x_train.index).drop(x_test.index).sample(frac=calib_fraction, random_state=s)
        
        y_train = y.loc[x_train.index]

        if max_normalization:
            scaling = np.mean(np.abs(y_train))
        else:
            scaling = 1

        y_train = y_train / scaling

        y_test = y.loc[x_test.index] / scaling

        y_calib = y.loc[x_calib.index] / scaling

        return x_train,y_train,x_calib,y_calib,x_test,y_test

    def simulate_data(self, cond_exp, noise_sd_fn, x_dist, n, p, T):
        n0 = int(n/2)
        n1 = n-n0-1
        data = generate_data(n, p, cond_exp, noise_sd_fn, x_dist)
        x = data[0]
                # for r in range(x.shape[0]):
                #     if x[r,0] < 0:
                #         x[r,1:] = x[r,0]

        y = data[1]

        if len(x.shape)==1:
            x = x.reshape(-1,1)

        x_train = x[:n0]
        y_train = y[:n0]

        x_calib = x[n0:n0+n1]
        y_calib = y[n0:n0+n1]



        test = generate_data(T, p, cond_exp, noise_sd_fn, x_dist)
        x_test = test[0]
                # for r in range(x.shape[0]):
                #     if x_even[r,0] < 0:
                #         x_even[r,1:] = x_even[r,0]

        y_test = test[1]
        return x_train,y_train,x_calib,y_calib,x_test,y_test

    def plot(self, metric='average_length_test', title_prefix=None, log_x=False, log_y=False, calc_only=False, 
             xlabel_conditional_coverage = "$X$", custom_title=None, ax=None, bigger_font=None, xlabel=None, ylabel=None):
        
        if self.local_metric:
            self.var_name = self.sorting_column
            df = self.results_df.copy()
        else:
            df = self.results_df.loc[self.results_df.metric == metric, ['UACQR-P', 'UACQR-S','CQR', 'CQR-r',self.var_name]]


        mean_results = df.groupby(self.var_name).mean()
        sem_results = df.groupby(self.var_name).sem()

        self.mean_results = mean_results
        self.sem_results = sem_results
        if calc_only:
            return

        if self.local_metric:
            self.var_name = xlabel_conditional_coverage

        if bigger_font:
            plt.rcParams['font.size'] = bigger_font
        else:
            plt.style.use(['default'])

        if not(ax):
            fig, ax = plt.subplots()

        if title_prefix:
            mean_results.plot(title=title_prefix+': '+ metric +' vs '+self.var_name+' ('+str(self.S)+' iters)', 
                              color=['tab:orange', 'tab:red','tab:green',  'tab:brown','tab:purple'], ax=ax)
        elif custom_title:
            mean_results.plot(title=custom_title, 
                              color=['tab:orange', 'tab:red','tab:green',  'tab:brown','tab:purple'], ax=ax)
        else:
            method_name = 'Fast UACQR-P' if self.fast_uacqr else 'Regular UACQR-P'
            mean_results.plot(title = method_name+': '+ metric +' vs '+self.var_name+' ('+str(self.S)+' iters)',
                                color=['tab:orange', 'tab:red', 'tab:green', 'tab:brown','tab:purple'], ax=ax)


        ax.fill_between(sem_results.index, mean_results['UACQR-P'] - 1.96 * sem_results['UACQR-P'],
                        mean_results['UACQR-P']+ 1.96 * sem_results['UACQR-P'],
                        color = 'tab:orange', alpha = .1)

        ax.fill_between(sem_results.index, mean_results['CQR'] - 1.96 * sem_results['CQR'],
                        mean_results['CQR'] + 1.96 * sem_results['CQR'],
                        color = 'tab:green', alpha = .1)

        ax.fill_between(sem_results.index, mean_results['UACQR-S'] - 1.96 * sem_results['UACQR-S'],
                        mean_results['UACQR-S'] + 1.96 * sem_results['UACQR-S'],
                        color = 'tab:red', alpha = .1)
        ax.fill_between(sem_results.index, mean_results['CQR-r'] - 1.96 * sem_results['CQR-r'],
                        mean_results['CQR-r'] + 1.96 * sem_results['CQR-r'],
                        color = 'tab:brown', alpha = .1)
        



        if self.local_metric:
            if metric == 'conditional_coverage':
                ax.axhline(y = 0.9, color = 'tab:blue', linestyle='--', zorder=0)
                ax.set_ylabel('Conditional Coverage')

            
            ax.fill_between(sem_results.index, mean_results['Base Estimator'] - 1.96 * sem_results['Base Estimator'],
                mean_results['Base Estimator'] + 1.96 * sem_results['Base Estimator'],
                color = 'tab:purple', alpha = .1)
            ax.set_xlabel(self.var_name)
            
        if xlabel:
            ax.set_xlabel(xlabel)
            
        if ylabel:
            ax.sex_ylabel(ylabel)


        if log_x:
            ax.xscale('log',  base=2)
        
        if log_y:
            ax.yscale('log',  base=2)
        


        


    def save(self, file_name):
        if not file_name.endswith('.pkl'):
            file_name += '.pkl'
        self.results_df.to_pickle(file_name)