#!/usr/bin/env python
# coding: utf-8

# # Dataset Preprocessing
# ## Import preprocessing libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import sklearn.datasets as ds
from torch import torch

# ## Preprocessing UCI

mat = pd.read_csv( "slump_test.data", header = 0, sep =",").values[:,1:]
np.save("UCI_dataset_01", np.delete(mat, -3, axis=1)[:,:-1])
np.save("UCI_dataset_02", np.delete(np.delete(mat, -2, axis=1) , -2, axis=1))

mat = pd.read_csv( "machine.data", header = None, sep =",").values[:,2:]
mat[mat == "?"] =None
mat = mat.astype(float)
mat[mat == None] = np.nan
mat = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(mat)
np.save("UCI_dataset_03", mat)

df = pd.read_excel( "2014 and 2015 CSM dataset.xlsx", header = 0)
Ratings_cols = ['Budget', 'Screens',
       'Sequel', 'Sentiment', 'Views', 'Likes', 'Dislikes', 'Comments',
       'Aggregate Followers', 'Ratings']
mat = df[Ratings_cols].values
mat = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(mat)
np.save("UCI_dataset_04", mat)

np.save("UCI_dataset_05", pd.read_csv( "yacht_hydrodynamics.data", header = None, sep="\s+").values)

np.save("UCI_dataset_06",pd.read_excel( "Concrete_Data.xls", header = 0).values.astype(float))

np.save("UCI_dataset_07", pd.read_csv("airfoil_self_noise.dat", header = None, sep ="\t").values)

mat = pd.read_csv( "communities.data", header = None, sep =",").values[:,5:]
mat[mat == "?"] =None
mat = mat.astype(float)
mat[mat == None] = np.nan
mat = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(mat)
np.save("UCI_dataset_08", mat)

np.save("UCI_dataset_09", pd.read_csv( "CASP.csv", header = 0)[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'RMSD']].values.astype(float))

np.save("UCI_dataset_10", pd.read_csv("Relation Network (Directed).data", header = None, sep =",").values[:,1:].astype(float))

cols = ['duration', 'width', 'height', 'bitrate', 'framerate',
       'i', 'p', 'b', 'frames', 'i_size', 'p_size', 'b_size', 'size',
        'o_bitrate', 'o_framerate', 'o_width', 'o_height', 'umem',
       'utime']

np.save("UCI_dataset_11", pd.read_csv("transcoding_mesurment.tsv" , header = 0, sep = "\t")[cols].values)

df = pd.read_csv("UJIndoorLoc/trainingData.csv" , header = 0, sep = ",").iloc[:,:522]
lat = [col for col in df.columns if col != 'LONGITUDE']
long = [col for col in df.columns if col != 'LATITUDE']
np.save("UCI_dataset_12", df[lat].values)
np.save("UCI_dataset_13", df[long].values)

np.save("UCI_dataset_14", pd.read_csv("slice_localization_data.csv" , header = 0, sep = ",").values[:,1:])

# ## Preprocessing news20

dataset_name = "news20"
X_train, y_train = ds.load_svmlight_file(dataset_name+'.scale', multilabel = False)
X_test, y_test = ds.load_svmlight_file(dataset_name+'.t.scale', multilabel = False)
    
cut_feature = 35830
#arbitrarly take 35830 first components to speed up computation time
X_train, X_test = X_train[:, :cut_feature], X_test[:, :cut_feature]

XT = X_train.transpose()
XTX = XT.dot(X_train)
dense_XTX = XTX.todense()
tensor_XTX = torch.tensor(dense_XTX).float()

# quite long process on cpu... (about 30 minutes)
with torch.no_grad():
    E, U = torch.symeig(tensor_XTX, eigenvectors=True)    
E = E.detach().numpy()
U = U.detach().numpy()
    
# keep 80 of variance by selecting the biggest eigen values
pos = E[E > 0.]
repartition = pos[::-1].cumsum()/pos.sum()
cut_eigen = 0.8
E_cut = E[-(repartition< cut_eigen).astype(int).sum():]
U_cut = U[:,-(repartition< cut_eigen).astype(int).sum():]
np.save(dataset_name+"_E_"+str(cut_eigen)+"_cut.npy",E_cut)
np.save(dataset_name+"_U_"+str(cut_eigen)+"_cut.npy",U_cut)

# transform features in principal components
np.save(dataset_name+"_X_train",X_train.dot(U_cut))
np.save(dataset_name+"_X_test",X_test.dot(U_cut))