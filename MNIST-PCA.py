# -*- coding: utf-8 -*-
"""
Parnyan Ataei
Applying PCA on MNIST datas 
"""
import numpy as np
import pandas as pd
import glob
import cv2 as cv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import svd

dir_address= 'E:/machine learning class/MNIST/trainingSet/*'
dir_items = []

features_pics = []
label_nums = []

for addr in glob.glob(dir_address):
    dir_items.append(addr[-1])

for item in dir_items: 
    for addr in glob.glob(dir_address+item+'/*'):
        img= cv.imread(addr).flatten()
        features_pics.append(img)
        label_nums.append(int(item))
        

scale = StandardScaler()
X_scaled = scale.fit_transform(features_pics)

# my PCA function
#def myPCA(X, k):
#    m= X.shape[0]
#    sigma = (X.T @ X) /m
#    U, S, _ = svd(sigma)
#    U_reduced = U[:, :k]
#    Z = X @ U_reduced
#    return Z
#
#pca= myPCA(X_scaled, 50)

pca = PCA()
X_scaled_pca= pca.fit_transform(X_scaled)
#X_scaled_pca= pd.DataFrame(X_scaled_pca)

explained_variance = pca.explained_variance_ratio_ 
print(explained_variance)  

