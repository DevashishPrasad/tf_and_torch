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
import numpy as np
from google.colab.patches import cv_imshow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Prepare Dataset

## Download and load dataset

mnist_train = torchvision.datasets.MNIST("/content/drive/My Drive/Siamese classifier", train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), target_transform=None, download=True)
mnist_test = torchvision.datasets.MNIST("/content/drive/My Drive/Siamese classifier", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=32, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=32, shuffle=True, num_workers=2)

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
num_epochs = 30

# No skip connections implemented
class SiaCNN(nn.Module):
  def __init__(self):
    super(SiaCNN, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(8),
      nn.ReLU(),
      nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU())      
    self.avgp = nn.AvgPool2d(14)
    
  def forward(self, x):
    x=x.float()
    out = self.layer1(x)
    # print(out.shape)
    out = self.avgp(out)
    out = out.reshape(out.size(0), -1)
    # print(out.shape)
    return out

model_ft = SiaCNN()

# Send the model to GPU
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.001)

decayRate = 0.99
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_ft, gamma=decayRate)

### Load again



# Training Pipeline

## Loss and Genearate pairs

## ============ old ======================
# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     """
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         return loss_contrastive
## ============ old ======================


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        # distances = np.linalg.norm(output2 - output1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

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

# # Verify Pair generation  
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

# for inputs, labels in dataloaders["train"]:
#   # print(inputs,labels)
#   # break
#   pairs = generate_pairs_from_batch(inputs,labels)
#   # print(pairs[0][2])
#   # i1 = [i[0] for i in pairs]
#   # i2 = [i[1] for i in pairs]
#   # lab = [i[2] for i in pairs]

#   # for i1,i2,lab in pairs:
#   #   i1,i2,lab = torch.tensor(i1).to(device),torch.tensor(i2).to(device),torch.tensor(lab).to(device)
#   #   print("PAIR : ",lab)
#   #   cv_imshow(np.array(i1).reshape(28,28)*255)
#   #   cv_imshow(np.array(i2).reshape(28,28)*255)
#   break  

def generate_triplets_from_batch(image_batch,label_batch):
  triplets = []
  for im1,b1 in zip(image_batch,label_batch):
    for im2,b2 in zip(image_batch,label_batch):
      if(torch.all(torch.eq(im1, im2))):
        continue
      if(b1==b2):
        for im3,b3 in zip(image_batch,label_batch):
          if(b1!=b3):
            triplets.append((im1,im2,im3))
  random.shuffle(triplets)
  return triplets

# # Verify Triplet generation  
# for inputs, labels in dataloaders["train"]:
#   # print(inputs,labels)
#   # break
#   triplets = generate_triplets_from_batch(inputs,labels)
#   print(len(triplets))
#   for i1,i2,i3 in triplets:
#     i1,i2,i3 = torch.tensor(i1).to(device),torch.tensor(i2).to(device),torch.tensor(i3).to(device)
#     cv_imshow(np.array(i1.cpu()).reshape(28,28)*255)
#     cv_imshow(np.array(i2.cpu()).reshape(28,28)*255)
#     cv_imshow(np.array(i3.cpu()).reshape(28,28)*255)
#     print("="*150)
#   break

## Training loop

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0

    for epoch in range(num_epochs):
        print('=' * 150)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            print('-' * 150)
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

                triplets = generate_triplets_from_batch(inputs,labels)
                no_ex += len(triplets)
                no_ex_itr += len(triplets)
                if(len(triplets)==0):
                  continue
                i1 = [] 
                i2 = []  
                i3 = []  
                for i in triplets:
                  i1.append(i[0].unsqueeze(0))
                for i in triplets:
                  i2.append(i[1].unsqueeze(0))
                for i in triplets:
                  i3.append(i[2].unsqueeze(0))

                i1 = torch.cat(i1).to(device)
                i2 = torch.cat(i2).to(device)
                i3 = torch.cat(i3).to(device)

                # for i1,i2,lab in pairs:
                #   i1,i2,lab = torch.tensor(i1).to(device),torch.tensor(i2).to(device),torch.tensor(lab).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    output1 = model(i1)
                    output2 = model(i2)
                    output3 = model(i3)
                    loss = criterion(output1, output2, output3)

                      # _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss
                loss_itr += loss
                if(ien%50==0):
                  lr = get_lr(optimizer) # latest pytorch 1.5+ uses get_last_lr,  previously it was get_lr iirc;
                  print('Phase[{}] Epoch {}/{} Iteration {}/{} :\t Total Loss: {:7f}  Iteration Loss: {:7f} Learning Rate : {:6f}'.format(phase, epoch, num_epochs - 1, ien, len(dataloaders[phase].dataset)/32,running_loss/no_ex,loss_itr/no_ex_itr,lr))
                  no_ex_itr = 0
                  my_lr_scheduler.step()
                  loss_itr = 0
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss/no_ex
            print('*********** Phase[{}] Epoch {}/{} :\t Epoch Loss: {:7f} ***********'.format(phase, epoch, num_epochs - 1, epoch_loss))

            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print(' {} Loss: {:.4f} '.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
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
            }, "/content/drive/My Drive/Siamese classifier/sia_mnist_new.pth")

    return model, val_loss_history, train_loss_history

# Run Training and Validation Step

from torchsummary import summary

summary(model_ft, (1, 28, 28))

# Setup the loss
# criterion = nn.CrossEntropyLoss()
# criterion = ContrastiveLoss()
criterion = TripletLoss()

# Train and evaluate
model_ft, val_hist, train_hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)

# Visualize the training

Ssad as  ddas d sa sdad aa aad s

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(train_hist)
plt.plot(val_hist)
plt.ylabel('Loss')
plt.savefig('/content/drive/My Drive/Siamese classifier/MNIST_64_sia_mnist_triplet.png')
plt.show()

# Testing
## Calculate avg distances in test set

from torch.autograd import Variable
chkpnt = torch.load("/content/drive/My Drive/Siamese classifier/sia_mnist.pth")
model_ft = SiaCNN()
model_ft.load_state_dict(chkpnt["model_state_dict"])

model_ft.eval()   # Set model to evaluate mode

final_true = 0
final_false = 0
no_batches = 0
# Verify Pair generation  
for inputs, labels in dataloaders["test"]:
  # print(inputs,labels)
  # break
  pairs = generate_pairs_from_batch(inputs,labels)
  # print(len(pairs))
  true_avg = 0
  false_avg = 0
  for en,(i1,i2,lab) in enumerate(pairs):
    i1 = torch.tensor(i1)
    i2 = torch.tensor(i2)
    lab = torch.tensor(lab)

    # print("PAIR : ",lab.data)
    emb1 = model_ft(i1.view(1,1,28,28))
    emb2 = model_ft(i2.view(1,1,28,28))
    # dist = F.pairwise_distance(emb1, emb2)
    dist = (emb1 - emb2).pow(2).sum(1)
    # print("DIST : ",dist)
    if(en <= len(pairs)/2):
      true_avg+=dist
    if(en > len(pairs)/2):
      false_avg+=dist      
    # cv_imshow(np.array(i1.cpu()).reshape(28,28)*255)
    # cv_imshow(np.array(i2.cpu()).reshape(28,28)*255)
  final_true += true_avg/len(pairs)/2  
  final_false += false_avg/len(pairs)/2  
  no_batches += 1
  print("True : ",true_avg/len(pairs)/2," False : ",false_avg/len(pairs)/2)
  # i1.cpu(),i2.cpu(),lab.cpu()
  del i1,i2,lab
  del inputs,labels
  del emb1,emb2
  del pairs
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()
  print("Final True : ",final_true/no_batches,"Final False : ",final_false/no_batches)

## Calculate accuracy of the classifier
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=1, shuffle=False, num_workers=2) 
from google.colab.patches import cv_imshow
import numpy as np  
chkpnt = torch.load("/content/drive/My Drive/Siamese classifier/sia_mnist.pth")
model_ft = SiaCNN()
model_ft.load_state_dict(chkpnt["model_state_dict"])

model_ft.eval()   # Set model to evaluate mode

trues = 0
total = 0
ground_embs = {}
count_arr = {}
for en,(i,lab) in enumerate(test_loader):
  emb1 = model_ft(i)
  if en < 200:
    if(en%20 == 0):
      print("*",end="")
    if(en%50 == 0):
      print()
    if int(lab.data[0]) not in ground_embs.keys():
      ground_embs[int(lab.data[0])] = emb1
      count_arr[int(lab.data[0])] = 1
    else:
      ground_embs[int(lab.data[0])] += emb1
      count_arr[int(lab.data[0])] += 1
  else:
    if(en == 200):
      print("\nNum of classes : ",len(ground_embs.keys()))
      for k in ground_embs.keys():
        print(count_arr[k])
        ground_embs[k] = ground_embs[k]/count_arr[k]
    if(en%50 == 0):
      print("=",end="")
    if(en%1000 == 0):
      print()
    total+=1
    for ge in ground_embs.keys():
      dist = (emb1 - ground_embs[ge]).pow(2).sum(1)
      if(dist < 3.3):
        if(ge == int(lab.data[0])):
          trues+=1
        break
print("\nTrue = ",trues," Total = ",total," Acc = ",trues/total)
