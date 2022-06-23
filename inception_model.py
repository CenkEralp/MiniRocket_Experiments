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

class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)
    
class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)

class InceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            #Reshape(out_shape=(1,160)),
            InceptionBlock(
                in_channels=1, 
                n_filters=32, 
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32*4, 
                n_filters=32, 
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32*4*1),
            nn.Linear(in_features=4*32*1, out_features=14)
        )
        
    def forward(self, xb):
        #print(xb.shape)
        return self.network(xb)