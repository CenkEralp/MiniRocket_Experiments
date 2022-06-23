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

class TimeSeriesCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # output: 64 x 125

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # output: 128 x 62

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # output: 256 x 31

            nn.Flatten(), 
            nn.Linear(4096, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 14))
        
    def forward(self, xb):
        return self.network(xb)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Create the MiniRocket features and store them in memory.
dsid = 'FacesUCR'
n_classes = 7
X, y, splits = get_UCR_data(dsid, split_data=False)

X_train = torch.tensor(np.squeeze(np.array(X))[splits[0]])
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val = torch.tensor(np.squeeze(np.array(X))[splits[1]])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
y_train = torch.tensor(y[splits[0]].astype(int)) - 1
y_val = torch.tensor(y[splits[1]].astype(int)) - 1

train_set = torch.utils.data.TensorDataset(X_train, y_train)
training_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True)

val_set = torch.utils.data.TensorDataset(X_val, y_val)
validation_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True, drop_last=True)

model = TimeSeriesCnnModel()
model.to(device)

# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
        outputs = model(inputs)

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

EPOCHS = 300

best_vloss = 1_000_000.
epoch_number = 0
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

for epoch in range(EPOCHS):
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    total_val_acc = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
        preds = torch.argmax(voutputs, 1) 
        total_val_acc += torch.sum(preds == vlabels) / len(vlabels)

    avg_vloss = running_vloss / (i + 1)
    avg_val_acc = total_val_acc / (i + 1)
    if epoch_number % 40 == 0:
      print('EPOCH {}:'.format(epoch_number + 1))
      print('LOSS train {} valid {} avg_val_acc: {}'.format(avg_loss, avg_vloss, avg_val_acc))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_best'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1