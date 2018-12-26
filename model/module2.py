import torch 
import torch.nn as nn
from inception import Inception
from interpolate import Interpolate

class Module2(nn.Module):
  def __init__(self, module_3):
    super(Module2, self).__init__()
    self.layer1 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2), # 4x
                    Inception(64, 16, [(3,16,16), (5,16,16), (7,16,16)]),
                    Inception(64, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    module_3,
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    Inception(128, 16, [(3,16,16), (5,16,16), (7,16,16)]),
                    Interpolate(scale_factor=2, mode='nearest') # up to 2x, output is 128 channel
                  )
    self.layer2 = nn.Sequential(
                    Inception(64, 16, [(3,16,16), (5,16,16), (7,16,16)]),
                    Inception(64, 16, [(3,32,16), (7,32,16), (11,32,16)])
                  )
      
  def forward(self,x):
    output1 = self.layer1(x)
    output2 = self.layer2(x)
    return output1 + output2
