import torch 
import torch.nn as nn
# For NormieNet
from module1 import Module1
from module2 import Module2
from module3 import Module3
from module4 import Module4

# Implementation of NormieNet: nickname for my altered version of the architecture published by Chen, et al. in NIPS 2016.
# Uses hourglass architecture
class NormieNet(nn.Module):
  def __init__(self):
    super(NormieNet, self).__init__()
    module_4 = Module4()
    module_3 = Module3(module_4)
    module_2 = Module2(module_3)
    module_1 = Module1(module_2)
    self.hourglass = nn.Sequential(
                      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
                      nn.BatchNorm2d(64),
                      nn.ReLU(),
                      module_1,
                      nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
                    )

  def forward(self, x):
    out = self.hourglass(x)
    return out
