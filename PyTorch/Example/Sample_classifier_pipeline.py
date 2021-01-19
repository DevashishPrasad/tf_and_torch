import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

### Prepare data 
images = [] #Images
labels = [] #Labels

### Visualize class balance
import numpy as np
un = np.unique(np.array(labels))

class_bal = []
for u in un:
  class_bal.append((np.array(labels)==u).sum())

class_bal.sort()

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.bar(range(0,3),class_bal, label="Number of images")
plt.ylabel('no of images')
plt.show()

### Data loaders

# Make the dataset mapping based for dataset loader
import torchvision
from PIL import Image

class SampleDataset(Dataset):
    def __init__(self):
        self.data = images
        self.target = labels
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        x = Image.fromarray(x)
        resizer = torchvision.transforms.Resize((50,50))
        tensor = torchvision.transforms.ToTensor()
        x = resizer(x)
        x = tensor(x)

        return x,y
    
    def __len__(self):
        return len(self.data)

b_size = 10

Edataset = SampleDataset()

print(len(Edataset)- 352)

# 15% Train - Test Split
train_set, test_set = torch.utils.data.random_split(Edataset, [2000, 352])

train_loader = DataLoader(train_set, batch_size=b_size, shuffle=True,num_workers=4)
test_loader = DataLoader(test_set, batch_size=b_size, shuffle=True,num_workers=4)

dataloaders = {'train':train_loader,'test':test_loader}

len(Edataset)

### Training pipeline

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('=' * 150)
        ep_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for ien, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                if(ien%5==0):
                  lr = get_lr(optimizer)
                  print('Phase[{}] Epoch {}/{} Iteration {}/{} :\t Epoch Loss: {:.10f} Accuracy: {:.10f} Learning Rate : {:.10f}'.format(phase, epoch, num_epochs - 1, ien, int(len(dataloaders[phase].dataset)/b_size),epoch_loss,epoch_acc,lr))
                if(ien%10==0 and phase == 'train'):
                  my_lr_scheduler.step()

            print('\n\n*********** Phase[{}] Epoch: {}/{} \t Epoch Acc: {:.10f} \t Epoch Loss: {:.10f} ***********\n\n'.format(phase, epoch, num_epochs - 1, epoch_acc, epoch_loss))

            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print(' {} Loss: {:.4f} '.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                # best_model_optm = copy.deepcopy(optimizer.state_dict())
                print("SAVING THE MODEL")
                # Save the best model
                torch.save(model, "/content/drive/My Drive/Orbit Shifters/trunk data/better_trunk_classifier.pth")
            if phase == 'test':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        est_time = ((time.time() - ep_time) / 60) * (num_epochs - epoch)
        print(" Estimated time remaining : {:.2f}m".format(est_time))

    time_elapsed = (time.time() - since)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # optimizer.load_state_dict(best_model_optm)

    # Save the best model
    # torch.save(model, "/content/drive/My Drive/Orbit Shifters/trunk data/trunk_classifier2.pth")

    return model, val_acc_history, train_acc_history


### build model

# No skip connections implemented
class Sample_Classifier(nn.Module):
  def __init__(self):
    super(Trunk_Classifier, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=3, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2))
    # Average pooling
    self.AvgPool = nn.AvgPool2d(11)
    self.fc1 = nn.Linear(64, 32)
    self.fc2 = nn.Linear(32, 3)

  def forward(self, x):
    x=x.float()
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.AvgPool(out)
    # print(out.shape)
    out = out.reshape(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    return out

model_ft = Sample_Classifier()

# Send the model to GPU
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)

decayRate = 0.985
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_ft, gamma=decayRate)

criterion = nn.CrossEntropyLoss()
from torchsummary import summary
import time

num_epochs = 30

summary(model_ft, (3,50,50))
model_ft, val_hist, train_hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.plot(train_hist)
plt.plot(val_hist)
plt.ylabel('accuracy')
plt.savefig('/content/drive/My Drive/Orbit Shifters/trunk data/model_acc.png')
plt.show()

from google.colab.patches import cv_imshow

tcm = torch.load("/content/drive/My Drive/Orbit Shifters/trunk data/better_trunk_classifier.pth").to(device)
tcm.eval()

total = 0
correct = 0
for im,l in test_loader:
  for i,la in zip(im,l):
    ggg = la.data
    # print("GT:",la.data)
    with torch.set_grad_enabled(False):
      pp = tcm(i.unsqueeze(0).cuda())
      pf = pp[0].tolist()
    ppp = pf.index(max(pf))
    # print("Pred:",)
    if(ppp == ggg):
      correct+=1
    total+=1

print(correct/total)
    # print("Pred:",pp.index(max(pp)))
    # cv_imshow(np.array(i.permute(1, 2, 0))*255)
