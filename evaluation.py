#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time as timer
from BKK_estimator import Closed_form_estimator
from ER_estimator import Closed_form_estimator as Closed_form_estimator0
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split as tts
import warnings

def ordered_train_test_split(X, y, n_split = 10, train_size = 0.8, random_state= 0):
    # cut observations in 10 deciles (in terms of y values), 
    # apply train test split on each decile,
    # gather back train and test from each decile
    ordered = np.argsort(y)
    l = int(len(y) / n_split)
    indexes = [np.argsort(y)[i*l:(i+1)*l] for i in range(n_split)]
    X_known, X_unknown, y_known, y_unknown = zip(*[tts(X[index], y[index], train_size = train_size, random_state = random_state+j) for j, index in enumerate(indexes)])
    X_known, X_unknown, y_known, y_unknown = np.concatenate( X_known, axis=0), np.concatenate( X_unknown, axis=0), np.concatenate( y_known, axis=0), np.concatenate( y_unknown, axis=0), 
    return X_known, X_unknown, y_known, y_unknown

def make_test(X_train, y_train, X_test, y_test, random_state = 0, method = "BKK", params = {}):
    warnings.filterwarnings("ignore")
    if method == "ER":
        time_start = timer()
        cfe = Closed_form_estimator0(eigen_decomposition=True, random_state = random_state, **params) 
        cfe = cfe.fit(X_train, y_train)
        res = cfe.score(X_test, y_test)
        time_stop = timer()
        return time_stop - time_start, res, cfe.old_loss, cfe.start_iter, np.exp(cfe._params["lambda"].detach().cpu().numpy())
    
    if method == "BKK":
        time_start = timer()
        cfe = Closed_form_estimator(eigen_decomposition=True, random_state = random_state, **params) 
        cfe = cfe.fit(X_train, y_train)
        res = cfe.score(X_test, y_test)
        time_stop = timer()
        return time_stop - time_start, res, cfe.old_loss, cfe.start_iter, np.exp(cfe._params["lambda"].detach().cpu().numpy())

    if method == "SBKK":
        time_start = timer()
        cfe = Closed_form_estimator(feature_sparsity=True, random_state = random_state, **params) 
        cfe = cfe.fit(X_train, y_train)
        res = cfe.score(X_test, y_test)
        time_stop = timer()
        return time_stop - time_start, res, cfe.old_loss, cfe.start_iter, np.exp(cfe._params["lambda"].detach().cpu().numpy())

    if method == "ABKK":
        time_start = timer()
        cfe = Closed_form_estimator(elastic_feature_sparsity=True, random_state = random_state, **params) 
        cfe = cfe.fit(X_train, y_train)
        res = cfe.score(X_test, y_test)
        time_stop = timer()
        return time_stop - time_start, res, cfe.old_loss, cfe.start_iter, np.exp(cfe._params["lambda"].detach().cpu().numpy())

    if method == "Ridge":
        time_start = timer()
        reg = RidgeCV(cv=5, fit_intercept = False, alphas = np.geomspace(1e-2,1e2,100), **params).fit(X_train, y_train)
        res = reg.score(X_test, y_test)
        time_stop = timer()
        return time_stop - time_start, res, 0, 0, reg.alpha_

    if method == "Lasso":
        time_start = timer()
        reg = LassoCV(cv=5, fit_intercept = False, **params).fit(X_train, y_train)
        res = reg.score(X_test, y_test)
        time_stop = timer()
        return time_stop - time_start, res, 0, reg.n_iter_, reg.alpha_

    if method == "Enet":
        time_start = timer()
        reg = ElasticNetCV(cv=5, fit_intercept = False, l1_ratio = [.1, .5, .7, .9, .95, .99, 1], **params).fit(X_train, y_train)
        res = reg.score(X_test, y_test)
        time_stop = timer()
        return time_stop - time_start, res, 0, reg.n_iter_, reg.alpha_


# # T experiment Synthetic

n_simulations = 100
n_features = 80
methods = ["ER","BKK"]
scenarii = ["A", "B", "C"]
dataset_repository = "dataset_folder/"
T_values = [0,1,3,10,30,100,300,1000]
first_time = False

if first_time: 
    Ts_recorded = np.zeros((len(T_values),len(scenarii), n_simulations, 5))
    Ts_processed = np.zeros((len(T_values),len(scenarii), n_simulations))
else:
    Ts_recorded = np.load(dataset_repository+"T_impact_synthetic_results.npy")
    Ts_processed = np.load(dataset_repository+"T_impact_synthetic_processed.npy")

i, method = 0, methods[0]
for j, scenario in enumerate(scenarii):
    dataset_name = dataset_repository+"synthetic_data_scenario_"+scenario+"_seed_"
    for k, seed in enumerate(range(n_simulations)):
        if Ts_processed[i,j,k] == 0.:
            try:
                print(i,j,k)
                X, y = np.load(dataset_name + str(seed)+"_X.npy"), np.load(dataset_name + str(seed)+"_y.npy")
                y = (y - y.mean()) / y.std()
                X_train, X_test, y_train, y_test = ordered_train_test_split(X, y, n_split = 10, train_size = 100/1100, random_state= seed*10)
                Ts_recorded[i,j,k] = np.array(make_test(X_train, y_train, X_test, y_test, random_state = seed, method = method))
                np.save(dataset_repository+"T_impact_synthetic_results", Ts_recorded)

                Ts_processed[i,j,k] = 1.
                np.save(dataset_repository+"T_impact_synthetic_processed", Ts_processed)
            except:
                print("error",i,j,k)

method = methods[1]
for i, T in enumerate(T_values)[1:]:
    BKK_params = {"n_permut":T}
    for j, scenario in enumerate(scenarii):
        dataset_name = dataset_repository+"synthetic_data_scenario_"+scenario+"_seed_"
        for k, seed in enumerate(range(n_simulations)):
            if Ts_processed[i,j,k] == 0.:
                try:
                    print(i,j,k)
                    X, y = np.load(dataset_name + str(seed)+"_X.npy"), np.load(dataset_name + str(seed)+"_y.npy")
                    y = (y - y.mean()) / y.std()
                    X_train, X_test, y_train, y_test = ordered_train_test_split(X, y, n_split = 10, train_size = 100/1100, random_state= seed*10)
                    Ts_recorded[i,j,k] = np.array(make_test(X_train, y_train, X_test, y_test, random_state = seed, method = method, params = BKK_params))
                    np.save(dataset_repository+"T_impact_synthetic_results", Ts_recorded)
                    
                    Ts_processed[i,j,k] = 1.
                    np.save(dataset_repository+"T_impact_synthetic_processed", Ts_processed)
                except:
                    print("error",i,j,k)


# # T experiment UCI small

n_simulations = 100
n_features = 80
methods = ["ER","BKK"]
scenarii = ["0" + val for val in np.arange(1,9).astype(str)]
dataset_repository = "dataset_folder/"
T_values = [0,1,3,10,30,100,300,1000]
first_time = False

if first_time: 
    Tu_recorded = np.zeros((len(T_values),len(scenarii), n_simulations, 5))
    Tu_processed = np.zeros((len(T_values),len(scenarii), n_simulations))
else:
    Tu_recorded = np.load(dataset_repository+"T_impact_UCI_results.npy")
    Tu_processed = np.load(dataset_repository+"T_impact_UCI_processed.npy")

i, method = 0, methods[0]
for j, scenario in enumerate(scenarii):
    dataset_name = dataset_repository+"UCI_dataset_"+scenario+".npy"
    X, y = np.load(dataset_name)[:,:-1],np.load(dataset_name)[:,-1]
    y = (y - y.mean()) / y.std()
    for k, seed in enumerate(range(n_simulations)):
        if Ts_processed[i,j,k] == 0.:
            try:
                print(i,j,k)
                X_train, X_test, y_train, y_test = ordered_train_test_split(X, y, n_split = 10, train_size = 0.8, random_state= seed*10)
                Tu_recorded[i,j,k] = np.array(make_test(X_train, y_train, X_test, y_test, random_state = seed, method = method))
                np.save(dataset_repository+"T_impact_UCI_results", Tu_recorded)

                Tu_processed[i,j,k] = 1.
                np.save(dataset_repository+"T_impact_UCI_processed", Tu_processed)
            except:
                print("error",i,j,k)

method = methods[1]
for i, T in enumerate(T_values)[1:]:
    BKK_params = {"n_permut":T}
    for j, scenario in enumerate(scenarii):
        dataset_name = dataset_repository+"UCI_dataset_"+scenario+".npy"
        X, y = np.load(dataset_name)[:,:-1],np.load(dataset_name)[:,-1]
        y = (y - y.mean()) / y.std()
        for k, seed in enumerate(range(n_simulations)):
            if Tu_processed[i,j,k] == 0.:
                try:
                    print(i,j,k)
                    X_train, X_test, y_train, y_test = ordered_train_test_split(X, y, n_split = 10, train_size = 0.8, random_state= seed*10)
                    Tu_recorded[i,j,k] = np.array(make_test(X_train, y_train, X_test, y_test, random_state = seed, method = method, params = BKK_params))
                    np.save(dataset_repository+"T_impact_UCI_results", Tu_recorded)
                    
                    Tu_processed[i,j,k] = 1.
                    np.save(dataset_repository+"T_impact_UCI_processed", Tu_processed)
                except:
                    print("error",i,j,k)


# # Synthetic data

n_simulations = 100
n_features = 80
methods = ["BKK", "SBKK", "ABKK","Ridge", "Lasso", "Enet"]
scenarii = ["A", "B", "C"]
dataset_repository = "dataset_folder/"
first_time = False

if first_time : 
    fs_processed = np.zeros((len(methods),len(scenarii), n_simulations))
    fs_recorded = np.zeros((len(methods),len(scenarii), n_simulations, 5))
else: 
    fs_processed = np.load(dataset_repository+"fast_synthetic_processed.npy")
    fs_recorded = np.load(dataset_repository+"fast_synthetic_results.npy")

for i, method in enumerate(methods):
    for j, scenario in enumerate(scenarii):
        dataset_name = dataset_repository+"synthetic_data_scenario_"+scenario+"_seed_"
        for k, seed in enumerate(range(n_simulations)):
            if fs_processed[i,j,k] == 0.:
                try:
                    print(i,j,k)
                    X, y = np.load(dataset_name + str(seed)+"_X.npy"), np.load(dataset_name + str(seed)+"_y.npy")
                    y = (y - y.mean()) / y.std()
                    X_train, X_test, y_train, y_test = ordered_train_test_split(X, y, n_split = 10, train_size = 100/1100, random_state= seed*10)
                    fs_recorded[i,j,k] = np.array(make_test(X_train, y_train, X_test, y_test, random_state = seed, method = method))
                    np.save(dataset_repository+"fast_synthetic_results", fs_recorded)
                    
                    fs_processed[i,j,k] = 1.
                    np.save(dataset_repository+"fast_synthetic_processed", fs_processed)
                except:
                    print("error",i,j,k)

# # 20news

scenarii = [500, 1000, 1500, 2000, 2500, 2875]
n_simulations = 100
methods = ["BKK", "SBKK", "ABKK", "Ridge", "Lasso", "Enet"]
dataset_repository = "dataset_folder/"
first_time = False

if first_time:
    svm_recorded = np.zeros((len(methods),len(scenarii), n_simulations, 5))
    svm_processed = np.zeros((len(methods),len(scenarii), n_simulations))
else:
    svm_recorded = np.load(dataset_repository+"fast_svmlib_results.npy")
    svm_processed = np.load(dataset_repository+"fast_svmlib_proccessed.npy")

X_train = np.load("news20__X_train_cut0.8.npy")
X_test = np.load("news20__X_test_cut0.8.npy")
y_train = np.load("news20_y_train.npy")
y_test = np.load("news20_y_test.npy")
m,s = y_train.mean(), y_train.std()
y_test = (y_test-m)/s
y_train = (y_train-m)/s

for i, method in enumerate(methods):
    for j, scenario in enumerate(scenarii):
        for k, seed in enumerate(range(n_simulations)):
            np.random.seed(k)
            sample = np.random.choice(np.arange(X_train.shape[0]), scenario, replace = False)
            _X_train, _y_train = X_train[sample], y_train[sample]
            _X_test, _y_test = X_test, y_test
            if svm_processed[i,j,k] == 0.:
                try:
                    print(i,j,k)
                    svm_recorded[i,j,k] = np.array(make_test(_X_train, _y_train, _X_test, _y_test, random_state = seed, method = method))
                    np.save(dataset_repository+"fast_svmlib_results", svm_recorded)
                    svm_processed[i,j,k] = 1.
                    np.save(dataset_repository+"fast_svmlib_processed", svm_processed)
                except:
                    print("error",i,j,k)


# # UCI

n_simulations = 100
methods = ["BKK", "SBKK", "ABKK", "Ridge", "Lasso", "Enet"]
scenarii = ["0" + val for val in np.arange(1,10).astype(str)]+list(np.arange(10,15).astype(str))
dataset_repository = "dataset_folder/"
first_time = False

if first_time:
    u_recorded = np.zeros((len(methods),len(scenarii), n_simulations, 5))
    u_processed = np.zeros((len(methods),len(scenarii), n_simulations))
else:
    u_recorded = np.load(dataset_repository+"fast_UCI_results.npy")
    u_processed = np.load(dataset_repository+"fast_UCI_proccessed.npy")

for j, scenario in enumerate(scenarii):
    dataset_name = dataset_repository+"UCI_dataset_"+scenario+".npy"
    X, y = np.load(dataset_name)[:,:-1],np.load(dataset_name)[:,-1]
    y = (y - y.mean()) / y.std()
    for i, method in enumerate(methods):
        for k, seed in enumerate(range(n_simulations)):
            if u_processed[i,j,k] == 0.:
                try:
                    print(i,j,k)
                    X_train, X_test, y_train, y_test = ordered_train_test_split(X, y, n_split = 10, train_size = 0.8, random_state= seed*10)
                    u_recorded[i,j,k] = np.array(make_test(X_train, y_train, X_test, y_test, random_state = seed, method = method))
                    np.save(dataset_repository+"fast_UCI_results", u_recorded)
                    u_processed[i,j,k] = 1.
                    np.save(dataset_repository+"fast_UCI_processed", u_processed)

                except:
                    print("error",i,j,k)