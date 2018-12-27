import torch 
import torch.nn as nn
from inception import Inception
from interpolate import Interpolate

class Module2(nn.Module):
  def __init__(self, module_3):
    super(Module2, self).__init__()
    self.layer1 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2), # 4x
                    Inception(128, 32, [(3,32,32), (5,32,32), (7,32,32)]),
                    Inception(128, 64, [(3,32,64), (5,32,64), (7,32,64)]),
                    module_3,
                    Inception(256, 64, [(3,32,64), (5,32,64), (7,32,64)]),
                    Inception(256, 32, [(3,32,32), (5,32,32), (7,32,32)]),
                    Interpolate(scale_factor=2, mode='nearest') # up to 2x, output is 128 channel
                  )
    self.layer2 = nn.Sequential(
                    Inception(128, 32, [(3,32,32), (5,32,32), (7,32,32)]),
                    Inception(128, 32, [(3,64,32), (7,64,32), (11,64,32)])
                  )
      
  def forward(self,x):
    output1 = self.layer1(x)
    output2 = self.layer2(x)
    return output1 + output2
