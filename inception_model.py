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
from sklearn import preprocessing

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
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model = nn.Sequential(
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
            nn.Linear(in_features=4*32*1, out_features=config["n_classes"])
        )
        
        #self.X, self.y, self.splits = get_UCR_data(config["dataset"], split_data=False)
        
    def forward(self, xb):
        return self.model(xb)

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        X, y, splits = get_UCR_data(self.config["dataset"], split_data=False)
        print(y)
        X_train = torch.tensor(np.array(X))[splits[0]]
        #X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = torch.tensor(np.array(X))[splits[1]]
        #X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        print(self.config["dataset"])
        le = preprocessing.LabelEncoder()
        le.fit(y)
        print(self.config["dataset"])
        y_train = torch.tensor(le.transform(y[splits[0]]))
        y_val = torch.tensor(le.transform(y[splits[1]]))
        print(X_train.shape, X_val.shape)
        train_set = torch.utils.data.TensorDataset(X_train, y_train)
        training_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True, drop_last=True)
        print(self.config["dataset"])
        val_set = torch.utils.data.TensorDataset(X_val, y_val)
        validation_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config["batch_size"], shuffle=True, drop_last=True)

        # Optimizers specified in the torch.optim package
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        loss_fn = torch.nn.CrossEntropyLoss()

        def train_one_epoch(epoch_index):
            running_loss = 0.
            last_loss = 0.

            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting
            for i, data in enumerate(training_loader):
                # Every data instance is an input + label pair
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = self.model(inputs)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 1000 == 999:
                    last_loss = running_loss / 1000 # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    tb_x = epoch_index * len(training_loader) + i + 1
                    running_loss = 0.

            return last_loss

        EPOCHS = self.config["epochs"]

        best_vloss = 1_000_000.
        epoch_number = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for epoch in range(EPOCHS):
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = train_one_epoch(epoch_number)

            # We don't need gradients on to do reporting
            self.model.train(False)

            running_vloss = 0.0
            total_val_acc = 0.0
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = self.model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                preds = torch.argmax(voutputs, 1) 
                total_val_acc += torch.sum(preds == vlabels) / len(vlabels)

            avg_vloss = running_vloss / (i + 1)
            avg_val_acc = total_val_acc / (i + 1)
            if epoch_number % 100 == 0:
                print('EPOCH {}:'.format(epoch_number + 1))
                print('LOSS train {} valid {} avg_val_acc: {}'.format(avg_loss, avg_vloss, avg_val_acc))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_best'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1
    def add_Inception_features(self, X, X_feat):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_output = torch.tensor(X).to(device)
        for i in range(len(self.model) - 1):
            x_output = self.model[i](x_output)
        print("X OUTPUT: ", x_output.cpu().detach().numpy().shape, X_feat.shape)
        result = np.hstack((X_feat, x_output.cpu().detach().numpy()))
        return result

    def load_model(self, model_path = "model_best"):
        self.model.load_state_dict(torch.load(model_path))
    
    def to_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
