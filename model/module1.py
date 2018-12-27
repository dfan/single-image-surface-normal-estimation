import torch 
import torch.nn as nn
from inception import Inception
from interpolate import Interpolate

class Module1(nn.Module):
  def __init__(self, module_2):
    super(Module1, self).__init__()
    self.layer1 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2), # 2x
                    Inception(128, 32, [(3,32,32), (5,32,32), (7,32,32)]),
                    Inception(128, 32, [(3,32,32), (5,32,32), (7,32,32)]),
                    module_2,
                    Inception(128, 32, [(3,64,32), (5,64,32), (7,64,32)]),
                    Inception(128, 16, [(3,32,16), (7,32,16), (11,32,16)]),
                    Interpolate(scale_factor=2, mode='nearest')
                  )
    self.layer2 = nn.Sequential(
                    Inception(128, 16, [(3,64,16), (7,64,16), (11,64,16)])
                  )
      
  def forward(self,x):
    output1 = self.layer1(x)
    output2 = self.layer2(x)
    return output1 + output2

