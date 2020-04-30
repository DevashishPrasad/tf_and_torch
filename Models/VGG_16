# VGG 16 architecture
class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),      
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),            
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer4 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),      
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),            
      nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer5 = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),      
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),            
      nn.MaxPool2d(kernel_size=2, stride=2))    
    # self.drop_out = nn.Dropout()
    self.fc1 = nn.Linear(6 * 6 * 512, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, 11)
    
  def forward(self, x):
    x=x.float()
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out
