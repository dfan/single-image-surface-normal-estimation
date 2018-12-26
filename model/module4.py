import torch 
import torch.nn as nn
from inception import Inception
from interpolate import Interpolate

class Module4(nn.Module):
  def __init__(self):
    super(Module4, self).__init__()
    self.layer1 = nn.Sequential(
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)])
                  )
    self.layer2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    Interpolate(scale_factor=2, mode='nearest') # Up to 8x, 256 channel
                  )
      
  def forward(self,x):
    output1 = self.layer1(x)
    output2 = self.layer2(x)
    return output1 + output2
