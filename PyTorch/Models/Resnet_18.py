class Res_Type(nn.Module):
  def __init__(self):
    super(Res_Type, self).__init__()
    
    # Commom Blocks 
    # Max Pool
    self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
    # Activation
    self.Activation = nn.ReLU()

    # Section 1 blocks
    self.layer1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(64),
      nn.ReLU(),      
      nn.MaxPool2d(kernel_size=3, stride=2))
    
    # Section 2 blocks
    # Normal block
    self.layer2_N = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64))
    # Identity block
    self.identity_2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)

    # Section 3 blocks
    # Initial block
    self.layer3_I = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128))
    # Normal block
    self.layer3_N = nn.Sequential(
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128))
    # Identity block
    self.identity_3 = nn.Conv2d(128, 256, kernel_size=1, stride=1)

    # Section 4 block
    # Initial block
    self.layer4_I = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256))
    # Normal block
    self.layer4_N = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256))      
    # Identity block
    self.identity_4 = nn.Conv2d(256, 512, kernel_size=1, stride=1)

    # Section 5 block
    # Initial block
    self.layer5_I = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512))
    # Normal block
    self.layer5_N = nn.Sequential(
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512))      

    # Average pooling
    self.AvgPool = nn.AvgPool2d(6)

    # Section 6 block
    self.layer6 = nn.Sequential(
      nn.Linear(512, 512),
      nn.Dropout(0.2),
      nn.Linear(512, 256),
      nn.Dropout(0.1),
      nn.Linear(256, 9),
      )
    
  def forward(self, x):
    x=x.float()

    # ==== section 1 (2 layers) ====
    op = self.layer1(x)
    # ==============================

    # ==== Section 2 (4 layers) ====
    x = self.layer2_N(op)
    x = self.Activation(x)
    x = self.layer2_N(x)
    sk = x + op # Skip connection normal
    x = self.Activation(sk)
    x = self.layer2_N(x)
    x = self.Activation(x)
    x = self.layer2_N(x)
    x = self.Activation(x)
    op = self.MaxPool(x)
    #===============================

    # ==== Section 3 (4 layers) ====
    sk = self.MaxPool(sk)
    sk = op + sk # Skip connection normal
    x = self.layer3_I(sk)
    x = self.Activation(x)
    x = self.layer3_N(x)
    sk = self.identity_2(sk)
    sk = sk + x # Skip connection dotted
    x = self.Activation(sk)
    x = self.layer3_N(x)
    x = self.Activation(x)
    x = self.layer3_N(x)
    x = self.Activation(x)
    op = self.MaxPool(x)
    #===============================
    
    # ==== Section 4 (4 layers) ====
    sk = self.MaxPool(sk)
    sk = op + sk # Skip connection normal
    x = self.layer4_I(sk)
    x = self.Activation(x)
    x = self.layer4_N(x)
    sk = self.identity_3(sk)
    sk = sk + x # Skip connection dotted
    x = self.Activation(sk)
    x = self.layer4_N(x)
    x = self.Activation(x)
    x = self.layer4_N(x)
    x = self.Activation(x)
    op = self.MaxPool(x)
    #===============================

    # ==== Section 5 (4 layers) ====
    sk = self.MaxPool(sk)
    sk = op + sk # Skip connection normal
    x = self.layer5_I(sk)
    x = self.Activation(x)
    x = self.layer5_N(x)
    sk = self.identity_4(sk)
    sk = sk + x # Skip connection dotted
    x = self.Activation(sk)
    x = self.layer5_N(x)
    x = self.Activation(x)
    x = self.layer5_N(x)
    op = self.Activation(x)
    #===============================

    # ==== Section 6 (3 layers) ====
    x = self.AvgPool(op)
    x = x.reshape(x.size(0), -1)
    out = self.layer6(x)  
    return out
