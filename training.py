from pickle import NONE
from tsai.basics import *
import sktime
import sklearn
from tsai.models.MINIROCKET import *
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.linear_model import RidgeClassifierCV
from datetime import datetime

from inception import Inception, InceptionBlock
from inception_model import InceptionModel

import catch22

class MiniRocketExperiment:
    def __init__(self, config, model = None):
        self.config = config
        self.model = model
    def train(self):
        X, y, splits = get_UCR_data(self.config["Dataset"], split_data=False)

        model_MiniRocket = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
        X_train = X[splits[0]]
        model_MiniRocket.fit(X_train)
        
        X_feat = get_minirocket_features(X, model_MiniRocket, chunksize=1024, to_np=True)
        X_feat = np.squeeze(np.array(X_feat))

        if self.config["Inception Features"]:
            print(X.shape)
            X_feat = self.model.add_Inception_features(X, X_feat)
        X = np.squeeze(np.array(X))

        if self.config["Mean"]:
            X_feat = np.hstack((X_feat, X.mean(axis = 1).reshape(-1, 1)))
        if self.config["Std"]:
            X_feat = np.hstack((X_feat, X.std(axis = 1).reshape(-1, 1)))
        if self.config["Max"]:
            X_feat = np.hstack((X_feat, X.max(axis = 1).reshape(-1, 1)))
        if self.config["Min"]:
            X_feat = np.hstack((X_feat, X.min(axis = 1).reshape(-1, 1)))
        if self.config["Catch22"]:
            catch22_features = []
            for i in range(len(X)):
                catch22_features.append(catch22.catch22_all(X[i])["values"])
            X_feat = np.hstack((X_feat,np.array(catch22_features)))

        if self.config["Square"]:
            X_feat = np.hstack((X_feat, X ** 2))
        if self.config["Cube"]:
            X_feat = np.hstack((X_feat, X ** 3))
        if self.config["Sin"]:
            X_feat = np.hstack((X_feat, np.sin(X)))
        if self.config["Cos"]:
            X_feat = np.hstack((X_feat, np.sin(X)))
        
        X_train2 = np.squeeze(np.array(X_feat))[splits[0]]
        X_val2 = np.squeeze(np.array(X_feat))[splits[1]]
        y_train2 = y[splits[0]]
        y_val2 = y[splits[1]]

        clf = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        print(X_train2.shape, y_train2.shape)
        clf.fit(X_train2, y_train2)
        print(clf.predict(X_train2).shape)
        print(clf.score(X_val2, y_val2))