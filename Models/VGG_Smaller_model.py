# Added Batch norm

class VGG_Type(nn.Module):
  def __init__(self):
    super(VGG_Type, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(v),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.fc1 = nn.Linear(25 * 25 * 128, 2048)
    self.fc2 = nn.Linear(2048, 1024)
    self.fc3 = nn.Linear(1024, 11)
    
  def forward(self, x):
    x=x.float()
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out
