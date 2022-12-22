import glob
import json
import logging
import os
#import tensorflow as tf
import torch
import tensorflow_datasets as tfds
import oskarsPackage.dn3_ext as dn3_ext
import numpy as np
from task2_regression.util.dataset_generator import RegressionDataGenerator, create_tf_dataset
import torch.optim as optim


def train_dataloader(dataset_train):
        y = tfds.as_numpy(dataset_train)
        return (y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m2 = dn3_ext.BENDRContextualizer(512,hidden_feedforward=3076)
m2.load_state_dict(torch.load("C:/Users/oskar/Downloads/contextualizer.pt",map_location=device))
#m2.double()
#print(m1.eval())
print(m2.eval())
for param in m2.parameters():
        param.requires_grad = False




window_length = 8 * 64  # 10 seconds
# Hop length between two consecutive decision windows
hop_length = 64
epochs = 100
patience = 5
batch_size = 8

















# THIS WHOLE BATCH SETS UP A DATA GENERATOR FOR MODELS.

# Get the path to the config gile
experiments_folder = os.path.dirname("C:/Users/oskar/Documents/GitHub/auditory-eeg-challenge-2023-code/task2_regression/experiments/vlaai.py")
task_folder = os.path.dirname(experiments_folder)
config_path = os.path.join(task_folder, 'util', 'config.json')
# Load the config
with open(config_path) as fp:
        config = json.load(fp)

data_folder = os.path.join(config["dataset_folder"], config["split_folder"])
stimulus_features = ["envelope"]
features = ["eeg"] + stimulus_features

train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # Create list of numpy array files
train_generator = RegressionDataGenerator(train_files, window_length)
dataset_train = create_tf_dataset(train_generator, window_length, None, hop_length, batch_size)

        # Create the generator for the validation set
val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
val_generator = RegressionDataGenerator(val_files, window_length)
dataset_val = create_tf_dataset(val_generator, window_length, None, hop_length, batch_size)

m1 = dn3_ext.ConvEncoderBENDR(20,encoder_h=512)
m1.load_state_dict(torch.load("C:/Users/oskar/Downloads/encoder.pt",map_location=device))
#m1 = m1.double()
for param in m1.parameters():
        param.requires_grad = False

class oskarNet(torch.nn.Module):
    def __init__(self, m1):
        super(oskarNet, self).__init__()
        self.pretrained = m1
        self.pretrained2 = m2
        self.my_new_layers = torch.nn.Sequential(torch.nn.Linear(7*512,1024),torch.nn.ReLU(),torch.nn.Linear(1024, 512))
    def forward(self, x):
        x = torch.as_tensor(x[:,:,0:20])
        x = torch.permute(x,(0,2,1))
        x = self.pretrained(x)
        x = self.pretrained2(x)
        x = torch.flatten(x,1)
        x = self.my_new_layers(x)
        #print(x.size())
        return x

def pearsons_loss(output, target):
        output = torch.unsqueeze(output,2)
        target = torch.unsqueeze(torch.as_tensor(target),2)
        C = torch.matmul(torch.permute(target-torch.mean(target,1,keepdim=True),(0,2,1)),output-torch.mean(output,1,keepdim=True))
        X = torch.matmul(torch.permute(target-torch.mean(target,1,keepdim=True),(0,2,1)),target-torch.mean(target,1,keepdim=True))
        Y = torch.matmul(torch.permute(output-torch.mean(output,1,keepdim=True),(0,2,1)),output-torch.mean(output,1,keepdim=True))
        loss = torch.div(C,torch.sqrt(torch.matmul(X,Y)))
        #print(loss)
        #print(loss.size())
        #loss = torch.corrcoef(torch.cat((target,output),1))
        loss = -torch.mean(loss)
        return loss

model = oskarNet(m1)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader(dataset_train), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, target = data
        #print(np.shape(inputs))
        #print(np.shape(target))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        print(outputs.size())
        loss = pearsons_loss(outputs, target[:,:,0])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 200:.3f}')
            running_loss = 0.0

for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_dataloader(dataset_val), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, target = data
        #print(np.shape(inputs))
        #print(np.shape(target))
        # zero the parameter gradients
        #optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        #loss = pearsons_loss(outputs, target[:,:,0])
        #loss.backward()
        #optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] val_loss: {running_loss / 2000:.3f}')
                running_loss = 0.0