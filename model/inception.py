import torch 
import torch.nn as nn

class Inception(nn.Module):
  def __init__(self, input_size, output_size, conv_params):
    super(Inception, self).__init__()
    # Base 1 x 1 conv layer
    self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=1, stride=1),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU()
                  )
    # Additional layer
    self.hidden = nn.ModuleList()
    for i in range(len(conv_params)):
      filt_size = conv_params[i][0]
      pad_size = int((filt_size - 1) / 2)
      out_a = conv_params[i][1]
      out_b = conv_params[i][2]
      curr_layer = nn.Sequential(
                    # Reduction
                    nn.Conv2d(in_channels=input_size, out_channels=out_a, kernel_size=1, stride=1),
                    nn.BatchNorm2d(out_a),
                    nn.ReLU(),
                    # Spatial convolution
                    nn.Conv2d(in_channels=out_a, out_channels=out_b, kernel_size=filt_size, stride=1, padding=pad_size),
                    nn.BatchNorm2d(out_b),
                    nn.ReLU()
                  )
      self.hidden.append(curr_layer)
      
  def forward(self,x):
    output1 = self.layer1(x)
    outputs = [output1]
    for i in range(len(self.hidden)):
      outputs.append(self.hidden[i](x))
    return torch.cat(outputs, 1)
