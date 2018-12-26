import torch
import torch.nn as nn

# Wrapper for nn.functional.interpolate to be used in nn.Sequential() module
class Interpolate(nn.Module):
  def __init__(self, size=None, scale_factor=None, mode=None, align_corners=None):
    super(Interpolate, self).__init__()
    self.size = size
    self.scale_factor = scale_factor
    self.mode = mode
    self.align_corners = align_corners
      
  def forward(self, x):
    x = nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners = self.align_corners)
    return x
