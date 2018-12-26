import torch 
import torch.nn as nn
from inception import Inception
from interpolate import Interpolate

class Module3(nn.Module):
  def __init__(self, module_4):
    super(Module3, self).__init__()
    self.layer1 = nn.Sequential(
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    Inception(128, 32, [(3,32,32), (7,32,32), (11,32,32)])
                  )
    self.layer2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2), # 8x
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    module_4, # down 16x then up to 8x
                    Inception(128, 32, [(3,16,32), (5,16,32), (7,16,32)]),
                    Inception(128, 32, [(3,32,32), (7,32,32), (11,32,32)]),
                    Interpolate(scale_factor=2, mode='nearest') # up to 4x. 256 channel
                  )
      
  def forward(self,x):
    output1 = self.layer1(x)
    output2 = self.layer2(x)
    return output1 + output2
