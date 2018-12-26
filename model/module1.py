import torch 
import torch.nn as nn
from inception import Inception
from interpolate import Interpolate

class Module1(nn.Module):
  def __init__(self, module_2):
    super(Module1, self).__init__()
    self.layer1 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2), # 2x
                    Inception(64, 16, [(3,16,16), (5,16,16), (7,16,16)]),
                    Inception(64, 16, [(3,16,16), (5,16,16), (7,16,16)]),
                    module_2,
                    Inception(64, 16, [(3,32,16), (5,32,16), (7,32,16)]),
                    Inception(64, 8, [(3,16,8), (7,16,8), (11,16,8)]),
                    Interpolate(scale_factor=2, mode='nearest') # up to original, 64 channel
                  )
    self.layer2 = nn.Sequential(
                    Inception(64, 8, [(3,32,8), (7,32,8), (11,32,8)])
                  )
      
  def forward(self,x):
    output1 = self.layer1(x)
    output2 = self.layer2(x)
    return output1 + output2

