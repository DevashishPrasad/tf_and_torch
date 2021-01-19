## Siamese network using contrastive loss MNIST dataset

# Import all libraries
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import copy
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare Dataset
## Download and load dataset
mnist_train = torchvision.datasets.MNIST("/content/drive/My Drive/Siamese classifier", train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), target_transform=None, download=True)
mnist_test = torchvision.datasets.MNIST("/content/drive/My Drive/Siamese classifier", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=8, shuffle=True)

dataloaders = {}
dataloaders["train"] = train_loader
dataloaders["test"] = test_loader

del mnist_train
del mnist_test

## Visualize some data

from google.colab.patches import cv_imshow
import numpy as np  

for i in train_loader:
  cv_imshow(np.array(i[0][0]).reshape(28,28)*255)

# Load Model

### create and load 

# Number of epochs to train for
num_epochs = 6

# No skip connections implemented
class SiaCNN(nn.Module):
  def __init__(self):
    super(SiaCNN, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),      
      nn.AvgPool2d(8))
    
  def forward(self, x):
    x=x.float()
    out = self.layer1(x)
    out = self.layer2(out)
    # print(out.shape)
    out = out.reshape(out.size(0), -1)
    # print(out.shape)
    return out

model_ft = SiaCNN()

# Send the model to GPU
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.0000001, momentum=0.9)

# Training Pipeline

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def generate_pairs_from_batch(image_batch,label_batch):
  pairs = []
  true_pairs = []
  false_pairs = []
  for im1,b1 in zip(image_batch,label_batch):
    for im2,b2 in zip(image_batch,label_batch):
      if(torch.all(torch.eq(im1, im2))):
        continue
      if(b1==b2):
        lab = 1
        true_pairs.append((im1,im2,lab))
      else:
        lab = 0
        false_pairs.append((im1,im2,lab))
  no_select = len(true_pairs)
  random.shuffle(false_pairs)

  pairs = true_pairs + false_pairs[:no_select]
  return pairs

# Verify Pair generation  
# for inputs, labels in dataloaders["test"]:
#   # print(inputs,labels)
#   # break
#   pairs = generate_pairs_from_batch(inputs,labels)
#   for i1,i2,lab in pairs:
#     i1,i2,lab = torch.tensor(i1).to(device),torch.tensor(i2).to(device),torch.tensor(lab).to(device)
#     print("PAIR : ",lab)
#     cv_imshow(np.array(i1).reshape(28,28)*255)
#     cv_imshow(np.array(i2).reshape(28,28)*255)
#   break

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0

    for epoch in range(num_epochs):
        print('=' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0
            no_ex = 0
            no_ex_itr = 0
            loss_itr = 0

            # Iterate over data.
            for ien, (inputs, labels) in enumerate(dataloaders[phase]):
                pairs = generate_pairs_from_batch(inputs,labels)
                no_ex += len(pairs)
                no_ex_itr += len(pairs)

                for i1,i2,lab in pairs:
                  i1,i2,lab = torch.tensor(i1).to(device),torch.tensor(i2).to(device),torch.tensor(lab).to(device)

                  # zero the parameter gradients
                  optimizer.zero_grad()

                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                      # Get model outputs and calculate loss
                      output1 = model(i1.view(1,1,28,28))
                      output2 = model(i2.view(1,1,28,28))
                      loss = criterion(output1, output2, lab)

                      # _, preds = torch.max(outputs, 1)

                      # backward + optimize only if in training phase
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()

                # statistics
                running_loss += loss
                loss_itr += loss
                if(ien%50==0):
                  print('Phase[{}] Epoch {}/{} Iteration {}/{} :\t Total Loss: {:7f}  Iteration Loss: {:7f} '.format(phase, epoch, num_epochs - 1, ien, len(dataloaders[phase].dataset)/8,running_loss/no_ex,loss_itr/no_ex_itr))
                  no_ex_itr = 0
                  loss_itr = 0
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss/no_ex
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print(' {} Loss: {:.4f} '.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                # best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model_optm = copy.deepcopy(optimizer.state_dict())
                # Save the best model
                torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, "/content/drive/My Drive/Siamese classifier/sia_mnist.pth")
            if phase == 'test':
                val_loss_history.append(running_loss/no_ex)
            if phase == 'train':
                train_loss_history.append(running_loss/no_ex)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    optimizer.load_state_dict(best_model_optm)

    # Save the best model
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, "/content/drive/My Drive/Siamese classifier/sia_mnist.pth")

    return model, val_loss_history, train_loss_history

# Run Training and Validation Step

from torchsummary import summary

summary(model_ft, (1, 28, 28))

# Setup the loss
# criterion = nn.CrossEntropyLoss()
criterion = ContrastiveLoss()

# Train and evaluate
model_ft, val_hist, train_hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)

# Visualize the training
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(val_hist)
plt.plot(train_hist)
plt.ylabel('accuracy')
plt.savefig('path to image.png')
plt.show()
