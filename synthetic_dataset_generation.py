#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.random import uniform, normal, choice, multivariate_normal
def generate_data(scenario, n, p, seed):
    np.random.seed(seed)
    if scenario == "A":
        noise = 50
        cov = (0.5**uniform(1,2, size = (p,p))) * (np.ones((p,p))-np.eye(p)) + np.eye(p)
        X = multivariate_normal(mean = np.zeros(p), cov = cov, size = n)
        coef = uniform(9,15, size = p) * np.sign(uniform(-1,1, size = p))
        y = X.dot(coef) + normal(scale = noise, size = n)
        return X, y       

    if scenario == "B":
        sparsity = 11 
        noise= 10
        X = normal(size = [n,p])
        coef = np.zeros(p)
        sup = choice(np.arange(p), sparsity, replace = False)
        coef[sup] = uniform(9,15, size = sparsity) * np.sign(uniform(-1,1, size = sparsity))
        y = X.dot(coef) + normal(scale = noise, size = n)
        return X, y
    
    if scenario == "C":
        sparsity = 35
        noise = 50
        cov = (0.5 ** uniform(1,2, size = (p,p))) * (np.ones((p,p))-np.eye(p)) + np.eye(p)
        X = multivariate_normal(mean = np.zeros(p), cov = cov, size = n)
        coef = np.zeros(p)
        coef[:sparsity] = uniform(9, 15, size = sparsity) * np.sign(uniform(-1,1, size = sparsity))
        y = X.dot(coef) + normal(scale = noise, size = n)
        return X, y

n_simulations = 100
n, p =  1000+100, 80
dataset_repository = "dataset_folder/"
for scenario in ["A","B","C"]:
    for seed in range(n_simulations):
        X, y = generate_data(scenario, n, p, seed)
        np.save(dataset_repository+"synthetic_data_scenario_"+scenario+"_seed_"+str(seed)+"_X", X)
        np.save(dataset_repository+"synthetic_data_scenario_"+scenario+"_seed_"+str(seed)+"_y", y)