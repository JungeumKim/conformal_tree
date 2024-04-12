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
from catboost import CatBoostRegressor

from helper import generate_data, interval_score_loss, randomized_conformal_cutoffs, select_column_per_row, average_coverage, average_interval_width
from helper import sample_binning_model, QuantileBandwidthModel, QuantileRegressionNN, corr_coverage_widths, hsic_coverage_widths, wsc
from helper import CatBoostWrapper



class uacqr():
    def __init__(self, model_params=dict(),
                 B=100, q_lower=5, q_upper=95, alpha=0.1, transform=None, inv_transform=None,
                 model_type='rfqr', bootstrapping_for_uacqrp=False, random_state=42,
                 uacqrs_agg='std', oracle_g=None, randomized_conformal=False,
                 subsample=1, sample_rfqr=False, extraneous_quantiles = ['mean'],
                 uacqrs_bagging=False, double_bootstrapping=False):
        '''
        inputs:
        model_params (dict): variables to pass into the model class
        B (int): number of bootstrapped runs or analogous for heuristic methods
        q_lower (float): target lower conditional quantile for quantile regression. Between 0 and 100. Default is 5
        q_upper (float): target upper conditional quantile for quantile regression. Between 0 and 100. Default is 95 
        alpha (float): target miscoverage rate, between 0 and 1
        model_type (str): type of model for quantile regression. Options: neural_net, rfqr, linear, lightgbm, knn, catboost
        random_state (int): random state for bootstrapping data and passed to model
        uacqrs_agg (str): either 'std' for standard deviation or 'iqr' for Innerquartile Range
        randomized_conformal (bool): use randomized conformal cutoffs to ensure exact 90% coverage
        extraneous_quantiles (list): if using neural_net, models these quantiles in addition to, e.g., 0.05 and 0.95. 'mean' uses MSE
        '''

        if transform is None:
            def transform(x):
                return x
        
        if inv_transform is None:
            def inv_transform(x):
                return x

        
        np.random.seed(random_state)
        quantiles=[q_lower, q_upper]

        if uacqrs_agg =='iqr':
            uacqrs_agg = iqr

        if (model_type == 'neural_net') and (model_params.get('dropout',0)==0) and not(model_params.get('epoch_model_tracking',False)):
            model_params["epoch_model_tracking"] = True
        elif (model_type == 'neural_net') and (model_params.get('dropout',0)>0) and not(model_params.get('epoch_model_tracking',False)):
            print("Will use dropout to generate B ensemble members. Set model_params[``epoch_model_tracking''] = True to use epoch-based heuristic instead")

        if q_upper <=1:
            print("Check units for q_upper, q_lower")
            q_lower = q_lower * 100
            q_upper = q_upper * 100

        self.model_params = model_params.copy()
        self.B = B
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.alpha = 0.1
        self.transform = transform
        self.inv_transform =  inv_transform
        self.model_type = model_type
        self.bootstrapping_for_uacqrp = bootstrapping_for_uacqrp
        self.random_state = random_state
        self.uacqrs_agg = uacqrs_agg
        self.oracle_g = oracle_g
        self.randomized_conformal = randomized_conformal
        self.subsample = subsample
        self.sample_rfqr = sample_rfqr
        self.extraneous_quantiles = extraneous_quantiles
        self.uacqrs_bagging = uacqrs_bagging
        self.double_bootstrapping = double_bootstrapping
        


    def fit(self, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train

        if not(self.bootstrapping_for_uacqrp) and (self.model_type not in ['rfqr','neural_net', 'catboost']):
            raise Exception("Cannot use Fast UACQR for models other than RFQR, CatBoost, and Neural Net")

        self.uacqrp_model_params = self.model_params.copy()
        if not self.bootstrapping_for_uacqrp and (self.model_type=='rfqr'):
            self.uacqrp_model_params["n_estimators"] = 1

            if not(self.double_bootstrapping):
                self.uacqrp_model_params["bootstrap"] = False

        self.models_B = []
        for b in range(self.B):
            b_idxs = np.random.choice(y_train.shape[0], size=int(y_train.shape[0]*self.subsample), replace=True)

            if isinstance(x_train, pd.DataFrame):
                x_train_b = x_train.iloc[b_idxs]
                y_train_b = y_train.iloc[b_idxs]
            else:
                x_train_b = x_train[b_idxs]
                y_train_b = y_train[b_idxs]
            
            
            if self.model_type=='rfqr':
                if self.sample_rfqr:
                    model_b = SampleRandomForestQuantileRegressor(q=[self.q_lower/100, self.q_upper/100], random_state=self.random_state+b, 
                                                        n_jobs=-1, **self.uacqrp_model_params)
                else:
                    model_b = RandomForestQuantileRegressor(q=[self.q_lower/100, self.q_upper/100], random_state=self.random_state+b, 
                                                        n_jobs=-1, **self.uacqrp_model_params)
                model_b.fit(x_train_b, y_train_b)
            elif self.model_type=='linear':
                model_b_lower = QuantileRegressor(quantile = self.q_lower/100, **self.uacqrp_model_params)
                model_b_upper = QuantileRegressor(quantile = self.q_upper/100, **self.uacqrp_model_params)
                model_b_lower.fit(x_train_b, y_train_b)
                model_b_upper.fit(x_train_b, y_train_b)
                model_b = [model_b_lower, model_b_upper]
            elif self.model_type=='lightgbm':
                model_b_lower = LGBMRegressor(objective='quantile', metric='quantile', boosting_type='gbdt',
                                            alpha=self.q_lower/100, n_jobs=-1, random_state=self.random_state+b, **self.uacqrp_model_params)
                model_b_upper = LGBMRegressor(objective='quantile', metric='quantile', boosting_type='gbdt',
                                            alpha=self.q_upper/100, n_jobs=-1, random_state=self.random_state+b, **self.uacqrp_model_params)
                model_b_lower.fit(x_train_b, y_train_b)
                model_b_upper.fit(x_train_b, y_train_b)
                model_b = [model_b_lower, model_b_upper]
            elif self.model_type=='sample_binning':
                model_b = sample_binning_model()

                model_b.fit(x_train_b, y_train_b)
            elif self.model_type == 'bandwidth':
                model_b = QuantileBandwidthModel(quantiles=[self.q_lower/100, self.q_upper/100], **self.uacqrp_model_params)
                model_b.fit(x_train_b, y_train_b)
            elif self.model_type == 'knn':
                model_b = KNeighborsQuantileRegressor(q=[self.q_lower/100, self.q_upper/100], **self.uacqrp_model_params)
                model_b.fit(x_train_b, y_train_b)
            elif (self.model_type == 'neural_net') and self.bootstrapping_for_uacqrp:
                self.modeled_quantiles = self.extraneous_quantiles.copy()
                self.modeled_quantiles.insert(0, self.q_lower/100)
                self.modeled_quantiles.append(self.self.q_upper/100)
                
                model_b = QuantileRegressionNN(quantiles=self.modeled_quantiles, 
                                                random_state=self.random_state+b, **self.uacqrp_model_params)
                model_b.fit(x_train_b, y_train_b)
            elif self.model_type =='neural_net' and not(self.bootstrapping_for_uacqrp):
                break
            elif self.model_type =='ls':
                model_b = LinearRegression(n_jobs=-1)
                model_b.fit(x_train_b, y_train_b)
                self.sigmahat = np.mean(np.square(model_b.predict(self,x_train)-self.y_train))**0.5
            elif self.model_type =='catboost' and self.bootstrapping_for_uacqrp:
                print("Not implemented yet to use boostrapping_for_uacqrp=True with CatBoost")
            elif self.model_type =='catboost' and not(self.bootstrapping_for_uacqrp):
                break

            
            self.models_B.append(model_b)
        
        if (self.model_type == 'neural_net') and not(self.bootstrapping_for_uacqrp):
            self.modeled_quantiles = self.extraneous_quantiles.copy()
            self.modeled_quantiles.insert(0, self.q_lower/100)
            self.modeled_quantiles.append(self.q_upper/100)
            
            nn_model = QuantileRegressionNN(quantiles=self.modeled_quantiles, 
                                            random_state=self.random_state, **self.model_params)
            nn_model.fit(x_train, y_train)

            self.nn_model = nn_model

        if (self.model_type == 'catboost') and not(self.bootstrapping_for_uacqrp):
            catboost_model_lower = CatBoostRegressor(loss_function=f'Quantile:alpha={self.q_lower/100}', random_state=self.random_state, 
                                                     verbose=False, **self.model_params)
            catboost_model_upper = CatBoostRegressor(loss_function=f'Quantile:alpha={self.q_upper/100}', random_state=self.random_state,
                                                     verbose=False, **self.model_params)

            catboost_model_lower.fit(x_train, y_train)
            catboost_model_upper.fit(x_train, y_train)

            self.catboost_model = [catboost_model_lower, catboost_model_upper]

            if self.B != catboost_model_lower.tree_count_:
                print("B != number of boosting rounds. Overriding to B = number of boosting rounds")
                self.B = min(catboost_model_lower.tree_count_, catboost_model_upper.tree_count_)
                
            if catboost_model_lower.tree_count_ != catboost_model_upper.tree_count_:
                print("Different number of boosting rounds for upper and lower quantile regressors. Overriding B to be the minimum of the two")

            self.models_B = [CatBoostWrapper(self.catboost_model, t) for t in range(self.B)]

        if not(self.bootstrapping_for_uacqrp):
            if self.model_type=='rfqr':
                self.model_params_cqr = self.model_params.copy()
                self.model_params_cqr['n_estimators'] =self.B
                if not(self.double_bootstrapping):
                    self.model_params_cqr["bootstrap"] = True
                self.cqr_base_model = RandomForestQuantileRegressor(q=[self.q_lower/100, self.q_upper/100], random_state=self.random_state, 
                                                        n_jobs=-1, **self.model_params_cqr)
                self.cqr_base_model.fit(x_train, y_train)
            elif self.model_type=='neural_net':
                self.cqr_base_model = self.nn_model
            elif self.model_type == 'catboost':
                self.cqr_base_model = CatBoostWrapper(self.catboost_model)
            else:
                print("model not implemented yet for non-median cqr")


    def calibrate(self, x_calib, y_calib, inject_noise=False, cond_exp=False, noise_sd_fn=False):
        n1 = x_calib.shape[0]
        
        calib_sorted_q_lower_ests = self.__predict_B_sorted(x_calib, lower = True)

        calib_sorted_q_lower_ests_df, scores_lower = self.__uacqrp_scores(y_calib, n1, calib_sorted_q_lower_ests, lower=True)
        
        calib_sorted_q_upper_ests = self.__predict_B_sorted(x_calib, lower = False)

        calib_sorted_q_upper_ests_df, scores_upper = self.__uacqrp_scores(y_calib, n1, calib_sorted_q_upper_ests, lower=False)
        
        if self.bootstrapping_for_uacqrp:
            cqr_scores_upper = self.transform(calib_sorted_q_upper_ests_df['truth']) - self.transform(calib_sorted_q_upper_ests_df[int(self.B/2)]) 

            cqr_scores_lower = self.transform(calib_sorted_q_lower_ests_df[int(self.B/2)])  - self.transform(calib_sorted_q_upper_ests_df['truth'])

            calib_interval_widths = self.transform(calib_sorted_q_upper_ests_df[int(self.B/2)]) - self.transform(calib_sorted_q_lower_ests_df[int(self.B/2)])
        else:           
            cqr_scores_upper = self.transform(calib_sorted_q_upper_ests_df['truth']) - self.transform(self.cqr_base_model.predict(x_calib)[-1]) 

            cqr_scores_lower = self.transform(self.cqr_base_model.predict(x_calib)[0])  - self.transform(calib_sorted_q_upper_ests_df['truth'])

            calib_interval_widths = self.transform(self.cqr_base_model.predict(x_calib)[-1]) - self.transform(self.cqr_base_model.predict(x_calib)[0])

        uacqrs_spread_upper = calib_sorted_q_upper_ests_df.iloc[:,:-2].agg(self.uacqrs_agg, axis=1) 
        uacqrs_spread_lower = calib_sorted_q_lower_ests_df.iloc[:,:-2].agg(self.uacqrs_agg, axis=1)

        if self.oracle_g:
            cqr_model_lower_calib = self.cqr_base_model.predict(x_calib)[0]
            cqr_model_upper_calib = self.cqr_base_model.predict(x_calib)[-1]

            true_quantile_lower = cond_exp(x_calib) - norm.ppf(self.q_upper/100, 0, noise_sd_fn(x_calib))
            true_quantile_upper = cond_exp(x_calib) + norm.ppf(self.q_upper/100, 0, noise_sd_fn(x_calib))

            uacqrs_spread_lower = np.abs(cqr_model_lower_calib - true_quantile_lower)
            uacqrs_spread_upper = np.abs(cqr_model_upper_calib - true_quantile_upper)

            self.cond_exp = cond_exp
            self.noise_sd_fn = noise_sd_fn
        
        if inject_noise:
            noise_upper_variance = calib_sorted_q_upper_ests_df.iloc[:,:-2].agg(self.uacqrs_agg, axis=1).var() * inject_noise
            noise_lower_variance = calib_sorted_q_lower_ests_df.iloc[:,:-2].agg(self.uacqrs_agg, axis=1).var() * inject_noise

            noise_upper = np.random.normal(size=calib_sorted_q_upper_ests_df.shape[0], scale=noise_upper_variance**0.5)
            noise_lower = np.random.normal(size=calib_sorted_q_lower_ests_df.shape[0], scale=noise_lower_variance**0.5)

            uacqrs_spread_upper = uacqrs_spread_upper + noise_upper
            uacqrs_spread_lower = uacqrs_spread_lower + noise_lower
                
            uacqrs_spread_upper.loc[uacqrs_spread_upper<0] = 0
            uacqrs_spread_lower.loc[uacqrs_spread_lower<0] = 0

        self.inject_noise = inject_noise



            

        if self.uacqrs_bagging:
            uacqrs_scores_upper = self.transform(calib_sorted_q_upper_ests_df['truth']) - self.transform(calib_sorted_q_upper_ests_df[int(self.B/2)])
            uacqrs_scores_lower = self.transform(calib_sorted_q_lower_ests_df[int(self.B/2)])  - self.transform(calib_sorted_q_upper_ests_df['truth'])
            uacqrs_scores_upper = uacqrs_scores_upper.div(uacqrs_spread_upper + 1e-05)
            uacqrs_scores_lower = uacqrs_scores_lower.div(uacqrs_spread_lower + 1e-05)
        else:
            uacqrs_scores_upper = cqr_scores_upper.div(uacqrs_spread_upper + 1e-05)
            uacqrs_scores_lower = cqr_scores_lower.div(uacqrs_spread_lower + 1e-05)
        
        cqrr_scores_upper = cqr_scores_upper.div(calib_interval_widths + 1e-05)
        cqrr_scores_lower = cqr_scores_lower.div(calib_interval_widths + 1e-05)
        
        scores = pd.DataFrame({"upper" : scores_upper, "lower" : scores_lower})
        
        cqr_scores = pd.DataFrame({"upper" : cqr_scores_upper, "lower" : cqr_scores_lower})

        uacqrs_scores = pd.DataFrame({"upper" : uacqrs_scores_upper, "lower" : uacqrs_scores_lower})
        
        cqrr_scores = pd.DataFrame({"upper" : cqrr_scores_upper, "lower" : cqrr_scores_lower})

        cqr_scores["combined"] = cqr_scores.max(axis=1)

        uacqrs_scores["combined"] = uacqrs_scores.max(axis=1)

        cqrr_scores["combined"] = cqrr_scores.max(axis=1)
        
        scores["combined"] = scores.max(axis=1)

        self.scores = scores
        self.cqr_scores = cqr_scores
        self.uacqrs_scores = uacqrs_scores
        self.cqrr_scores = cqrr_scores

        self.score_threshold = scores["combined"].sort_values(ascending=True).iloc[math.ceil((1-self.alpha)*(n1+1))-1]
        self.cqr_score_threshold = cqr_scores["combined"].sort_values(ascending=True).iloc[math.ceil((1-self.alpha)*(n1+1))-1]
        self.uacqrs_score_threshold = uacqrs_scores["combined"].sort_values(ascending=True).iloc[math.ceil((1-self.alpha)*(n1+1))-1]
        self.cqrr_score_threshold = cqrr_scores["combined"].sort_values(ascending=True).iloc[math.ceil((1-self.alpha)*(n1+1))-1]

        if self.randomized_conformal:
            print("Warning: score threshold attributes at this point are not randomized. They will be after making predictions")



    def predict(self, x_test):
        if len(x_test.shape)==1:
            x_test = x_test.reshape(-1,1)
        
        if self.randomized_conformal:
            self.score_threshold = randomized_conformal_cutoffs(self.scores['combined'], x_test.shape[0], alpha=self.alpha).astype(int)
            self.cqr_score_threshold = randomized_conformal_cutoffs(self.cqr_scores['combined'], x_test.shape[0], alpha=self.alpha)
            self.uacqrs_score_threshold = randomized_conformal_cutoffs(self.uacqrs_scores['combined'], x_test.shape[0], alpha=self.alpha)
            self.cqrr_score_threshold = randomized_conformal_cutoffs(self.cqrr_scores['combined'], x_test.shape[0], alpha=self.alpha)

        q_lower_uacqrp = self.__predict_B_sorted(x_test, lower = True)
        q_upper_uacqrp = self.__predict_B_sorted(x_test, lower=False)

        uacqrs_spread_upper_test = pd.DataFrame(q_upper_uacqrp).iloc[:,:-1].agg(self.uacqrs_agg, axis=1) 
        uacqrs_spread_lower_test = pd.DataFrame(q_lower_uacqrp).iloc[:,:-1].agg(self.uacqrs_agg, axis=1)

        if self.oracle_g:
            cqr_model_lower = self.cqr_base_model.predict(x_test)[0]
            cqr_model_upper = self.cqr_base_model.predict(x_test)[-1]

            true_quantile_lower = self.cond_exp(x_test) - norm.ppf(self.q_upper/100, 0, self.noise_sd_fn(x_test))
            true_quantile_upper = self.cond_exp(x_test) + norm.ppf(self.q_upper/100, 0, self.noise_sd_fn(x_test))

            uacqrs_spread_lower_test = pd.Series(np.abs(cqr_model_lower - true_quantile_lower))
            uacqrs_spread_upper_test = pd.Series(np.abs(cqr_model_upper - true_quantile_upper))

        if self.inject_noise:
            noise_upper_variance = uacqrs_spread_upper_test.var() * self.inject_noise
            noise_lower_variance = uacqrs_spread_lower_test.var() * self.inject_noise

            noise_upper = np.random.normal(size=uacqrs_spread_upper_test.shape[0], scale=noise_upper_variance**0.5)
            noise_lower = np.random.normal(size=uacqrs_spread_lower_test.shape[0], scale=noise_lower_variance**0.5)

            uacqrs_spread_upper_test = uacqrs_spread_upper_test + noise_upper
            uacqrs_spread_lower_test = uacqrs_spread_lower_test + noise_lower
                
            uacqrs_spread_upper_test.loc[uacqrs_spread_upper_test<0] = 0
            uacqrs_spread_lower_test.loc[uacqrs_spread_lower_test<0] = 0


        if self.bootstrapping_for_uacqrp:
            cqr_model_lower = q_lower_uacqrp[:,int(self.B/2)]
            cqr_model_upper = q_upper_uacqrp[:,int(self.B/2)]
        else:
            cqr_model_lower = self.cqr_base_model.predict(x_test)[0]
            cqr_model_upper = self.cqr_base_model.predict(x_test)[-1]    
        
        test_interval_widths = cqr_model_upper - cqr_model_lower


        self.test_y_lower = select_column_per_row(q_lower_uacqrp, self.score_threshold)
        self.test_y_lower_cqr = self.inv_transform(self.transform(cqr_model_lower) - self.cqr_score_threshold)
        self.test_y_lower_uacqrs = self.inv_transform(self.transform(cqr_model_lower) - self.uacqrs_score_threshold*(uacqrs_spread_lower_test.values + 1e-05))
        self.test_y_lower_cqrr = self.inv_transform(self.transform(cqr_model_lower) - self.cqrr_score_threshold*(test_interval_widths + 1e-05))
        self.test_y_lower_median = q_lower_uacqrp[:,int(self.B/2)]
        self.test_y_lower_base = cqr_model_lower

        self.test_y_upper = select_column_per_row(q_upper_uacqrp, self.score_threshold)
        self.test_y_upper_cqr = self.inv_transform(self.transform(cqr_model_upper) + self.cqr_score_threshold)
        self.test_y_upper_uacqrs = self.inv_transform(self.transform(cqr_model_upper) + self.uacqrs_score_threshold*(uacqrs_spread_upper_test.values + 1e-05))
        self.test_y_upper_cqrr = self.inv_transform(self.transform(cqr_model_upper) + self.cqrr_score_threshold*(test_interval_widths + 1e-05))
        self.test_y_upper_median = q_upper_uacqrp[:,int(self.B/2)]
        self.test_y_upper_base = cqr_model_upper

        bounds_df = pd.DataFrame(data={"UACQR-P_lower":self.test_y_lower, "UACQR-P_upper":self.test_y_upper,
                                       "UACQR-S_lower":self.test_y_lower_uacqrs, "UACQR-S_upper":self.test_y_upper_uacqrs,
                                       "CQR_lower":self.test_y_lower_cqr, "CQR_upper":self.test_y_upper_cqr,
                                       "CQR-r_lower":self.test_y_lower_cqrr, "CQR-r_upper":self.test_y_upper_cqrr,
                                       "Base_lower":self.test_y_lower_base, "Base_upper":self.test_y_upper_base})
        bounds_df.columns = pd.MultiIndex.from_product((['UACQR-P','UACQR-S','CQR','CQR-r','Base'],['lower','upper']))

        return bounds_df

    def evaluate(self, x_test, y_test, oqr_metrics=False): 
        
        self.x_test = x_test
        self.y_test = y_test

        bounds_df = self.predict(x_test)
        methods = ['UACQR-P','UACQR-S','CQR','CQR-r','Base']
        metrics = dict()

        metrics["interval_score_loss"] = {method: interval_score_loss(bounds_df[method]["upper"].values, bounds_df[method]["lower"].values, 
                                                                      y_test, alpha=self.alpha).mean() for method in methods}

        metrics["test_coverage"] = {method: average_coverage(bounds_df[method]["upper"].values, bounds_df[method]["lower"].values, 
                                                                      y_test) for method in methods}
        
        metrics["average_length_test"] = {method: average_interval_width(bounds_df[method]["upper"].values, bounds_df[method]["lower"].values) 
                                          for method in methods}
        
        if oqr_metrics:
            metrics["oqr_corr"] = {method: corr_coverage_widths(bounds_df[method]["upper"].values, bounds_df[method]["lower"].values, 
                                                                      y_test) for method in methods}
            metrics["oqr_hsic"] = {method: hsic_coverage_widths(bounds_df[method]["upper"].values, bounds_df[method]["lower"].values, 
                                                                      y_test) for method in methods}
            
            metrics["oqr_wsc"] = {method: wsc(x_test, y_test, bounds_df[method]["upper"].values, bounds_df[method]["lower"].values) 
                                          for method in methods}

        for outer_key, inner_dict in metrics.items():
            for inner_key, inner_value in inner_dict.items():
                setattr(self, f"{inner_key.replace('-','').lower()}_{outer_key}", inner_value)

        return metrics        






    def calculate_conditional_coverage(self, cond_exp=None, noise_sd_fn=None, plot=True, sorting_column=0, round_to=2, title=None,
                                        included_methods=['UACQR-P','UACQR-S','CQR', 'CQR-r','Base Estimator'],
                                        column_by_feature_importance=False, ecdf=False, metric='conditional_coverage'):
        
        if not(hasattr(self,"test_cover_rates")) and cond_exp and metric=='conditional_coverage':
            miscover_below_rate = norm.cdf(self.test_y_lower, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))   
            miscover_below_rate_cqr = norm.cdf(self.test_y_lower_cqr, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            miscover_below_rate_uacqrs = norm.cdf(self.test_y_lower_uacqrs, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            miscover_below_rate_cqrr = norm.cdf(self.test_y_lower_cqrr, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            if not(self.bootstrapping_for_uacqrp):
                miscover_below_rate_base = norm.cdf(self.test_y_lower_base, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            else:
                miscover_below_rate_base = norm.cdf(self.test_y_lower_median, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))

            miscover_above_rate = 1 - norm.cdf(self.test_y_upper, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            miscover_above_rate_cqr = 1 - norm.cdf(self.test_y_upper_cqr, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            miscover_above_rate_uacqrs = 1 - norm.cdf(self.test_y_upper_uacqrs, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            miscover_above_rate_cqrr = 1 - norm.cdf(self.test_y_upper_cqrr, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            if not(self.bootstrapping_for_uacqrp):
                miscover_above_rate_base = 1 - norm.cdf(self.test_y_upper_base, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))
            else:
                miscover_above_rate_base = 1 - norm.cdf(self.test_y_upper_median, loc = cond_exp(self.x_test), scale=noise_sd_fn(self.x_test))


            self.test_cover_rates = 1 - miscover_below_rate - miscover_above_rate
            self.test_cover_rates_cqr = 1 - miscover_below_rate_cqr - miscover_above_rate_cqr
            self.test_cover_rates_uacqrs = 1 - miscover_below_rate_uacqrs - miscover_above_rate_uacqrs
            self.test_cover_rates_cqrr = 1 - miscover_below_rate_cqrr - miscover_above_rate_cqrr
            self.test_cover_rates_base = 1 - miscover_below_rate_base - miscover_above_rate_base
        else:
            if metric == 'conditional_coverage':
                self.test_cover_rates = (self.test_y_lower<=self.y_test)&(self.test_y_upper>=self.y_test)
                self.test_cover_rates_cqr = (self.test_y_lower_cqr<=self.y_test)&(self.test_y_upper_cqr>=self.y_test)
                self.test_cover_rates_uacqrs = (self.test_y_lower_uacqrs<=self.y_test)&(self.test_y_upper_uacqrs>=self.y_test)
                self.test_cover_rates_cqrr = (self.test_y_lower_cqrr<=self.y_test)&(self.test_y_upper_cqrr>=self.y_test)
                self.test_cover_rates_base = (self.test_y_lower_base<=self.y_test)&(self.test_y_upper_base>=self.y_test)
            elif metric == 'average_length_test':
                self.test_cover_rates = self.test_y_upper - self.test_y_lower
                self.test_cover_rates_cqr = self.test_y_upper_cqr - self.test_y_lower_cqr
                self.test_cover_rates_uacqrs = self.test_y_upper_uacqrs - self.test_y_lower_uacqrs
                self.test_cover_rates_cqrr = self.test_y_upper_cqrr - self.test_y_lower_cqrr
                self.test_cover_rates_base = self.test_y_upper_base - self.test_y_lower_base



        x_df = pd.DataFrame(self.x_test)
        
        if cond_exp:
            x_df[sorting_column] = x_df[sorting_column].round(round_to)
        else:
            if column_by_feature_importance:
                sorting_column = np.argmax(self.self.cqr_base_model.feature_importances_)
            if ecdf:
                x_df[sorting_column] = x_df[sorting_column].rank(pct=True).round(round_to)
            else:
                x_df[sorting_column] = x_df[sorting_column].rank(pct=True).round(round_to)

        x_df['UACQR-P'] = self.test_cover_rates
        x_df['UACQR-S'] = self.test_cover_rates_uacqrs
        x_df['CQR'] = self.test_cover_rates_cqr
        x_df['CQR-r'] = self.test_cover_rates_cqrr
        x_df['Base Estimator'] = self.test_cover_rates_base

        if not(title):
            title = 'Conditional Coverage'

        self.conditional_coverage = x_df.groupby(sorting_column)[included_methods].mean()

        if plot:
            self.conditional_coverage.plot(title=title,
                                            color=['tab:orange', 'tab:red', 'tab:green', 'tab:brown','tab:purple'])
            if metric == 'condtional_coverage':
                plt.axhline(y = 0.9, color = 'tab:blue', linestyle='--', alpha=1)  
            
            plt.xlabel('$X_{\\cdot,'+str(sorting_column+1)+'}$')
                                                      


    def plot(self, cond_exp, noise_sd_fn, feature_for_x_axis=0):
        colors = {'UACQR-P':'tab:orange', 'CQR':'tab:green', 'UACQR-S':'tab:red', 'Base Estimator':'tab:purple', 
          'CQR-m with oracle info':'tab:brown','UACQR-S with oracle info':'tab:pink', 'Truth':'tab:blue',
          'Median Estimator':'tab:purple', 'RFQR':'tab:purple', 'CQR-r':'tab:brown'}
        
        idxs = self.x_test[:,feature_for_x_axis].argsort()
        
        x_column = self.x_test[idxs,feature_for_x_axis]

        fig, ax = plt.subplots()
        ax.scatter(x_column, self.y_test[idxs], color='0.5', alpha=0.4)

        ax.plot(x_column, (cond_exp(self.x_test) - norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
            label='Truth', linewidth=3, color=colors['Truth'])
        ax.plot(x_column, self.test_y_lower[idxs], label='UACQR-P',
                alpha=1, color=colors['UACQR-P'])
        ax.plot(x_column, self.test_y_lower_cqr[idxs], label='CQR',
                alpha=1, color=colors['CQR'])
        ax.plot(x_column, self.test_y_lower_uacqrs[idxs], label='UACQR-S',
                alpha=1, color=colors['UACQR-S'])
        ax.plot(x_column, self.test_y_lower_cqrr[idxs], label='CQR-r',
                alpha=1, color=colors['CQR-r'])
        if self.bootstrapping_for_uacqrp:
            ax.plot(x_column, self.test_y_lower_median[idxs], label='Median Estimator',
                alpha=1, color=colors['Median Estimator'])
        else:
            ax.plot(x_column, self.test_y_lower_base[idxs], label='Base Estimator',
                alpha=1, color=colors['Base Estimator'])


        ax.plot(x_column, (cond_exp(self.x_test) + norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
            label='Truth', linewidth=3)
        ax.plot(x_column, self.test_y_upper[idxs], label='UACQR-P',
                alpha=1, color=colors['UACQR-P'])
        ax.plot(x_column, self.test_y_upper_cqr[idxs], label='CQR',
                alpha=1, color=colors['CQR'])
        ax.plot(x_column, self.test_y_upper_uacqrs[idxs], label='UACQR-S',
                alpha=1, color=colors['UACQR-S'])
        ax.plot(x_column, self.test_y_upper_cqrr[idxs], label='CQR-r',
                alpha=1, color=colors['CQR-r'])
        if self.bootstrapping_for_uacqrp:
            ax.plot(x_column, self.test_y_upper_median[idxs], label='Median Estimator',
                alpha=1, color=colors['Median Estimator'])
        else: 
            ax.plot(x_column, self.test_y_upper_base[idxs], label='Base Estimator',
                alpha=1, color=colors['Base Estimator'])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        

    def plot_simple(self, cond_exp, noise_sd_fn, feature_for_x_axis=0, sharey=True, suptitle=None,
                    ylabel = '$Y$', xlabel = '$X$', expanded=False):
        colors = {'UACQR-P':'tab:orange', 'CQR':'tab:green', 'UACQR-S':'tab:red', 'Base Estimator':'tab:purple', 
          'CQR-m with oracle info':'tab:brown','UACQR-S with oracle info':'tab:pink', 'Truth':'tab:blue',
          'Median Estimator':'tab:purple', 'RFQR':'tab:purple', 'CQR-r':'tab:brown'}
        
        idxs = self.x_test[:,feature_for_x_axis].argsort()
        idxs_train = self.x_train[:,feature_for_x_axis].argsort()
        
        x_column = self.x_test[idxs,feature_for_x_axis]
        x_train_column = self.x_train[idxs_train,feature_for_x_axis]

        plt.rcParams['font.size'] = 18
        if expanded:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=sharey, figsize=(15,12))
            plt.setp(((ax1, ax2), (ax3, ax4)), ylim=(-4,4))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=sharey, figsize=(10,3))
            plt.setp((ax1, ax2), ylim=(-4,4))
        
        fig.subplots_adjust(top=0.75)
        fig.suptitle(suptitle, fontsize=13)

        ax1.scatter(x_train_column, self.y_train[idxs_train], color='0.1', alpha=1, s=10, label='Training Samples')
        ax1.fill_between(x_column, (cond_exp(self.x_test) - norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
                          (cond_exp(self.x_test) + norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
            label='Oracle', color=colors['Truth'], alpha=0.3)
        ax1.fill_between(x_column, self.test_y_lower_cqr[idxs],  self.test_y_upper_cqr[idxs], label='CQR',
                color=colors['CQR'], alpha=0.3)

        
        ax1.legend(loc='upper left')
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel)
        ax1.set_title('Existing method: CQR')


        ax2.scatter(x_train_column, self.y_train[idxs_train], color='0.1', alpha=1, s=10, label='Training Samples')
        ax2.fill_between(x_column, (cond_exp(self.x_test) - norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
                          (cond_exp(self.x_test) + norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
            label='Oracle',color=colors['Truth'], alpha=0.3)
        ax2.fill_between(x_column, self.test_y_lower[idxs], self.test_y_upper[idxs], label='UACQR-P',
                color=colors['UACQR-P'], alpha=0.3)
        
        ax2.legend(loc='upper left')
        ax2.set_xlabel(xlabel)
        ax2.set_title('Proposal: UACQR-P')


        if expanded:
            ax3.scatter(x_train_column, self.y_train[idxs_train], color='0.1', alpha=1, s=10, label='Training Samples')
            ax3.fill_between(x_column, (cond_exp(self.x_test) - norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
                            (cond_exp(self.x_test) + norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
                label='Oracle', color=colors['Truth'], alpha=0.3)
            ax3.fill_between(x_column, self.test_y_lower_cqrr[idxs],  self.test_y_upper_cqrr[idxs], label='CQR-r',
                    color=colors['CQR-r'], alpha=0.3)

            
            ax3.legend(loc='upper left')
            ax3.set_ylabel(ylabel)
            ax3.set_xlabel(xlabel)
            ax3.set_title('Existing method: CQR-r')
            
            
            ax4.scatter(x_train_column, self.y_train[idxs_train], color='0.1', alpha=1, s=10, label='Training Samples')
            ax4.fill_between(x_column, (cond_exp(self.x_test) - norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
                            (cond_exp(self.x_test) + norm.ppf(self.q_upper/100, 0, noise_sd_fn(self.x_test)))[idxs],
                label='Oracle',color=colors['Truth'], alpha=0.3)
            ax4.fill_between(x_column, self.test_y_lower_uacqrs[idxs], self.test_y_upper_uacqrs[idxs], label='UACQR-S',
                    color=colors['UACQR-S'], alpha=0.3)
            
            ax4.legend(loc='upper left')
            ax4.set_xlabel(xlabel)
            ax4.set_title('Proposal: UACQR-S')
        plt.rcParams['font.size'] = 10


    def __predict_individual_model(self, x_calib, model,lower=True):
        if lower:
            i = 0
        else:
            i = -1
        if self.model_type=='rfqr':
            return model.predict(x_calib)[i]
        elif self.model_type=='catboost':
            return model.predict(x_calib)[i]
        elif self.model_type=='linear':
            return model[i].predict(x_calib)
        elif self.model_type=='lightgbm':
            return model[i].predict(x_calib)
        elif self.model_type=='sample_binning' or self.model_type=='bandwidth' or self.model_type=='knn':
            return model.predict(x_calib)[i]
        elif (self.model_type == 'neural_net') and self.bootstrapping_for_uacqrp:
            return model.predict(x_calib)[i]
        elif self.model_type == 'ls' and lower:
            return model.predict(x_calib) + norm.ppf(self.q_lower/100) * self.sigmahat
        elif self.model_type == 'ls' and not(lower):
            return model.predict(x_calib) + norm.ppf(self.q_upper/100) * self.sigmahat
        

    def __predict_B_sorted(self, x_test, lower=True):
        q_lower_uacqrp = np.zeros((len(x_test),self.B+1))
        for t,model in enumerate(self.models_B):
            q_lower_uacqrp[:,t] = self.__predict_individual_model(x_test, model, lower=lower)
        
        if self.model_type == 'neural_net' and not(self.bootstrapping_for_uacqrp):
            q_lower_uacqrp = self.nn_model.predict(x_test, ensembling=self.B+1)[0 if lower else -1]
        
        q_lower_uacqrp[:,self.B] = -np.inf if lower else np.inf
        q_lower_uacqrp.sort(axis=1) 
        if lower:
            q_lower_uacqrp = q_lower_uacqrp[:,::-1]
        return q_lower_uacqrp
    
    def __uacqrp_scores(self, y_calib, n1, calib_sorted_ests, lower=True):
        calib_sorted_q_ests_df = pd.DataFrame(data=calib_sorted_ests, index=range(n1), columns=range(self.B+1))

        if isinstance(y_calib, pd.Series):
            calib_sorted_q_ests_df['truth'] = y_calib.values
        else:
            calib_sorted_q_ests_df['truth'] = y_calib

        def find_minimal_bound(series):
            if lower:
                return (series <= series['truth']).idxmax()
            else:
                return (series >= series['truth']).idxmax()

        scores = calib_sorted_q_ests_df.apply(find_minimal_bound, axis=1, raw=False)
        return calib_sorted_q_ests_df, scores






        
