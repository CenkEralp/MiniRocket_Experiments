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
import itertools
 

class MiniRocketExperiment:
    def __init__(self, model = None):
        self.model = model
    def train(self, config):
        X, y, splits = get_UCR_data(config["Dataset"], split_data=False)

        model_MiniRocket = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
        X_train = X[splits[0]]
        model_MiniRocket.fit(X_train)
        
        X_feat = get_minirocket_features(X, model_MiniRocket, chunksize=1024, to_np=True)
        X_feat = np.squeeze(np.array(X_feat))

        if config["Inception Features"]:
            X_feat = self.model.add_Inception_features(X, X_feat)
        X = np.squeeze(np.array(X))

        if config["Mean"]:
            X_feat = np.hstack((X_feat, X.mean(axis = 1).reshape(-1, 1)))
        if config["Std"]:
            X_feat = np.hstack((X_feat, X.std(axis = 1).reshape(-1, 1)))
        if config["Max"]:
            X_feat = np.hstack((X_feat, X.max(axis = 1).reshape(-1, 1)))
        if config["Min"]:
            X_feat = np.hstack((X_feat, X.min(axis = 1).reshape(-1, 1)))
        if config["Catch22"]:
            catch22_features = []
            for i in range(len(X)):
                catch22_features.append(catch22.catch22_all(X[i])["values"])
            X_feat = np.hstack((X_feat,np.array(catch22_features)))

        if config["Square"]:
            X_feat = np.hstack((X_feat, X ** 2))
        if config["Cube"]:
            X_feat = np.hstack((X_feat, X ** 3))
        if config["Sin"]:
            X_feat = np.hstack((X_feat, np.sin(X)))
        if config["Cos"]:
            X_feat = np.hstack((X_feat, np.sin(X)))
        
        X_train2 = np.squeeze(np.array(X_feat))[splits[0]]
        X_val2 = np.squeeze(np.array(X_feat))[splits[1]]
        y_train2 = y[splits[0]]
        y_val2 = y[splits[1]]

        clf = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        #print(X_train2.shape, y_train2.shape)
        clf.fit(X_train2, y_train2)
        #print(clf.predict(X_train2).shape)
        return clf.score(X_val2, y_val2)

    def perform_all_tests(self):
        datasets = get_UCR_univariate_list()
        results = []
        len_datasets = len(datasets)
        for i, dataset in enumerate(datasets):
            tests = list(itertools.product([False, True], repeat=4))
            result = [] 
            for i, test in enumerate(tests):
                Experiment_config = {"Dataset": "MedicalImages", "Inception Features": test[0],
                                    "Mean": test[1], "Std": test[1], "Max": test[1], "Min": test[1],
                                    "Catch22": test[2], "Square": test[3], "Cube": test[3], "Sin": test[3],
                                    "Cos": test[3]}
                acc = self.train(Experiment_config)
                result.append(acc)
            best_acc = max(result)
            #print("Experiment {}/{}: {} Normal acc: {} Best acc: {} Best Experiment: {}".format(i+1, len_datasets, dataset, result[0], best_acc, result.index(best_acc)))
            print(str(result), ",")
        return np.array(results)


def get_UCR_univariate_list():
    return [
        'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
        'AllGestureWiimoteZ', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken',
        'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration',
        'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
        'CricketZ', 'Crop', 'DiatomSizeReduction',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxTW', 'DodgerLoopDay', 'DodgerLoopGame',
        'DodgerLoopWeekend', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
        'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal',
        'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
        'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain',
        'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3',
        'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint', 'GunPointAgeSpan',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham',
        'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
        'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
        'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
        'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
        'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
        'MoteStrain', 'NonInvasiveFetalECGThorax1',
        'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
        'PhalangesOutlinesCorrect', 'Phoneme', 'PickupGestureWiimoteZ',
        'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane',
        'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
        'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
        'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
        'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
        'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
        'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
        'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
        'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
        'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
        'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
    ]