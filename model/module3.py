import torch 
import torch.nn as nn
from inception import Inception
from interpolate import Interpolate

class Module3(nn.Module):
  def __init__(self, module_4):
    super(Module3, self).__init__()
    self.layer1 = nn.Sequential(
                    Inception(256, 64, [(3,32,64), (5,32,64), (7,32,64)]),
                    Inception(256, 64, [(3,64,64), (7,64,64), (11,64,64)])
                  )
    self.layer2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2), # 8x
                    Inception(256, 64, [(3,32,64), (5,32,64), (7,32,64)]),
                    Inception(256, 64, [(3,32,64), (5,32,64), (7,32,64)]),
                    module_4, # down 16x then up to 8x
                    Inception(256, 64, [(3,32,64), (5,32,64), (7,32,64)]),
                    Inception(256, 64, [(3,64,64), (7,64,64), (11,64,64)]),
                    Interpolate(scale_factor=2, mode='nearest') # up to 4x. 256 channel
                  )
      
  def forward(self,x):
    output1 = self.layer1(x)
    output2 = self.layer2(x)
    return output1 + output2
