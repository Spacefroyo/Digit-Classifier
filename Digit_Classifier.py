# import math
# arr = [ 1.2641e+00,  2.0243e+00, -4.4311e-01,  4.3837e-01,  9.4906e-01,
#            -7.9173e-01,  1.0709e+00,  2.1515e+00, -1.7070e+00,  2.9910e-01]
# print(sum([math.e**x for x in arr]))
# exit()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import prod

import torch
from torch import flatten
from torch.nn import Linear, Module, Conv2d, LogSoftmax, ModuleList, Softmax
from torch.nn.functional import relu, cross_entropy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from mnist import MNIST


# CLASSIFIER

class DigitClassifier(Module):
    # convs = number of channels
    def __init__(self, input_size: tuple, input_channels: int, classes: int, convs: list[int], fcs: list[int], kernel = (5, 5), padding = (0, 0)):
        super().__init__()
        convs = [input_channels] + convs
        self.convs = ModuleList([Conv2d(convs[i], convs[i+1], kernel) for i in range(len(convs)-1)])

        flattened_size = prod([input_size[i]-(kernel[i]-1+padding[i]*2)*(len(convs)-1) for i in range(len(input_size))])
        fcs = [convs[-1]*flattened_size] + fcs + [classes]
        self.fcs = ModuleList([Linear(fcs[i], fcs[i+1]) for i in range(len(fcs)-1)])

        # print(self.modules)
        # exit()

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
            x = relu(x)

        x = flatten(x, 1)

        for layer in self.fcs[:-1]:
            x = layer(x)
            x = relu(x)
        x = self.fcs[-1](x)

        output = LogSoftmax(dim=1)(x)

        return output



def compute_accuracy(model, dataloader, device):

    correct, total = 0, 0
    for idx, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device).tolist()

        pred = model(image).tolist()
        
        arr = [max(range(len(pred[i])), key=pred[i].__getitem__) == label[i] for i in range(len(pred))]
        correct += sum(arr)
        total += len(arr)
    
    return correct/total

def compute_council_accuracy(models, dataloader, device):

    correct, total = 0, 0
    for idx, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device).tolist()

        preds = [model(image).tolist() for model in models]
        
        arr = [max(arr:=[max(range(len(pred[i])), key=pred[i].__getitem__) for pred in preds], key=arr.count) == label[i] for i in range(len(label))]
        correct += sum(arr)
        total += len(arr)
    
    return correct/total


# DATA

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.005
CLASSES = 10
INPUT_SIZE, INPUT_CHANNELS = (28, 28), 1

CONVS, FCS = [32], [128, 64]

mndata = MNIST('MNIST')
mndata.gz = False
trainset = mndata.load_training()
trainset = pd.DataFrame([(trainset[1][i], trainset[0][i]) for i in range(len(trainset[0]))])

testset = mndata.load_testing()
testset = pd.DataFrame([(testset[1][i], testset[0][i]) for i in range(len(testset[0]))])

class FormatDataFrame(Dataset) :

  def __init__(self, df) :
    self.df = df

  def __len__(self) :
    return len(self.df)

  def __getitem__(self, index) :
    return self.df.iloc[index]

def vectorize_batch(batch) :

  Y = tuple([data[0] for data in batch])
  X = [np.reshape(np.array(data[1]), (INPUT_CHANNELS, *INPUT_SIZE)) for data in batch]

  return torch.tensor(np.array(X), dtype=torch.float), torch.tensor(Y)

trainloader = DataLoader(FormatDataFrame(trainset), batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch)
testloader = DataLoader(FormatDataFrame(testset), batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorize_batch)


# MODELS

import copy

def train(num_models, device, do_print=True):
    models, accs = [None] * num_models, [0] * num_models
    for i in range(num_models):
        print("\n\nMODEL", i+1)
        model = DigitClassifier(INPUT_SIZE, INPUT_CHANNELS, CLASSES, CONVS, FCS)
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

        losses, train_accs, test_accs = [], [], []

        import time

        startTime = time.time()
        for epoch in range(EPOCHS):
            model.train()
            for idx, (image, label) in enumerate(trainloader):
                image, label = image.to(device), label.to(device)

                pred = model(image)
                loss = cross_entropy(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not idx%100:
                    losses.append(loss.item())
                    if do_print:
                        print(f'Epoch: {epoch + 1:03d}/{EPOCHS:03d} | '
                        f'Batch: {idx:03d}/{len(trainloader):03d} | '
                        f'Loss:  {loss:.4f}')

            train_acc = compute_accuracy(model, trainloader, device)
            test_acc = compute_accuracy(model, testloader, device)
            if do_print:
                print(f'training accuracy: '
                    f'{train_acc*100:.2f}%'
                    f'\ntest accuracy: '
                    f'{test_acc*100:.2f}%')
                print(f'Total elapsed: {(time.time() - startTime)/60:.2f} min\n')

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if (accs[i] < test_acc):
                models[i] = copy.deepcopy(model)
                accs[i] = test_acc

    return models, accs

import os

_, _, files = next(os.walk("classifier_models/"))
TRAINED_MODELS = len(files)

NUM_MODELS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = "classifier_models/model"
RETRAIN = False
if RETRAIN:
    models, accs = train(NUM_MODELS, device, False)
else:
    models, accs = [], []
    for i in range(min(TRAINED_MODELS, NUM_MODELS)):
        model = DigitClassifier(INPUT_SIZE, INPUT_CHANNELS, CLASSES, CONVS, FCS)
        model.load_state_dict(torch.load(PATH+str(i)))
        model.eval()

        models.append(model)
        accs.append(compute_accuracy(model, testloader, device))

    if TRAINED_MODELS < NUM_MODELS:
        models_append, accs_append = train(NUM_MODELS-TRAINED_MODELS, device, False)
        models.extend(models_append)
        accs.extend(accs_append)

for i in range(0 if RETRAIN else TRAINED_MODELS, len(models)):
    torch.save(models[i].state_dict(), PATH+str(i))

print("Individual accuracies: ", accs)

council_acc = compute_council_accuracy(models, testloader, device)
print("Council Accuracy: ", council_acc)