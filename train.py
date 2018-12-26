import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from imageio import imread, imwrite
import numpy as np
import os
import sys
sys.path.append('model')
from model import *

# Dataset subclass for loading images efficiently in memory
# Assumes data is structured as follows: a "train" folder and "test" folder in "data"
# train has a "color", "mask", and "normal" folder. test has only a "color" and "mask" folder
class NormalsDataset(data.dataset.Dataset):
  def __init__(self, is_train, transform):
    self.is_train = is_train
    self.transform = transform
  def __len__(self):
    if self.is_train:
      return 20000 # 20000 training images
    return 2000 # 2000 testing images

  def __getitem__(self, index):
    data_type = 'train' if self.is_train else 'test'
    image = np.array(imread('data/{}/color/{}.png'.format(data_type, index))) # 128 x 128 x 3
    if self.transform:
      image = self.transform(image) # ToTensor converts (HxWxC) -> (CxHxW)
    mask = np.array(imread('data/{}/mask/{}.png'.format(data_type, index))) # 128 x 128
    mask = torch.from_numpy(mask) / 255 # binary 0 and 1s

    if not self.is_train:
      return index, image, mask

    normal = np.array(imread('data/{}/normal/{}.png'.format(data_type, index))) # 128 x 128 x 3
    if self.transform:
      normal = self.transform(normal) * 2 - 1 # get normals in range -1 to 1
    return index, image, mask, normal

def get_loss(preds, truths):
  # Calculate loss : average cosine value between predicted/actual normals at each pixel
  # theta = arccos((P dot Q) / (|P|*|Q|)) -> cos(theta) = (P dot Q) / (|P|*|Q|)
  # Both the predicted and ground truth normals normalized to be between -1 and 1
  preds_norm =  torch.nn.functional.normalize(preds, p=2, dim=1)
  truths_norm = torch.nn.functional.normalize(truths, p=2, dim=1)
  # make negative so function decreases (cos -> 1 if angles same)
  loss = -torch.sum(preds_norm * truths_norm, dim = 1)
  return loss

def evaluate(model, device, data_loader):
  model.eval() # set model to evaluation mode (VERY IMPORTANT)
  with torch.no_grad():
    # Measure: mean cosine angle error over all pixels
    mean_angle_error = 0
    total_pixels = 0
    for _, images, masks, normals in data_loader:
      images = images.to(device)         # batch_size x 3 x 128 x 128
      masks = masks.to(device)           # batch_size x 128 x 128
      ground_truths = normals.to(device) # batch_size x 3 x 128 x 128

      preds = model(images)
    
      # Rearrange outputs to batch_size x 128 x 128 x 3 to apply masks
      # Output is now _ x 3 (rows of length 3 vectors)
      preds = preds.permute(0,2,3,1)[masks,:]
      truths = ground_truths.permute(0,2,3,1)[masks,:]

      loss = get_loss(preds, truths)
      mean_angle_error += torch.sum(loss)
      total_pixels += loss.numel()
    return mean_angle_error / total_pixels

###
### Input/output directories
###
MODEL_DIR = 'model.ckpt'
TEST_PREDS_DIR = 'test_prediction'   # where to put predictions on testing set
TRAIN_PREDS_DIR = 'train_prediction' # where to put predictions on training set (for debugging)

def train(finetune, finetune_epochs):
  # Hyper parameters
  num_epochs = 75
  learning_rate = 0.001

  train_params = {'batch_size': 25, 'shuffle': True, 'num_workers': 5}
  test_params = {'batch_size': 50, 'shuffle': True, 'num_workers': 5}

  # Load Data
  train_set = NormalsDataset(is_train = True, transform=transforms.ToTensor())
  test_set = NormalsDataset(is_train = False, transform=transforms.ToTensor())
  train_loader = data.DataLoader(train_set, **train_params)
  test_loader = data.DataLoader(test_set, **test_params)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = NormieNet().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  # Load existing model and do finetuning. Only on GPU
  if finetune:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, nesterov=True)
    train_params = {'batch_size': 50, 'shuffle': True, 'num_workers': 5}
    train_loader = data.DataLoader(train_set, **train_params)
    model.load_state_dict(torch.load(MODEL_DIR))
    model = model.to(device)
    num_epochs = finetune_epochs    

  # Train the network
  total_steps = len(train_loader)
  iterations = []
  losses = []
  accuracy = []
  for epoch in range(num_epochs):
    # Assuming batch size of 25 and learning rate 0.001 with Adam prior
    if epoch == 17 and not finetune:
      optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
      # Increase batch size during slow learning phase
      train_params = {'batch_size': 50, 'shuffle': True, 'num_workers': 5}
      train_loader = data.DataLoader(train_set, **train_params)
    elif epoch == 27 and not finetune:
      optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True)
    elif epoch == 47 and not finetune:
      optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, nesterov=True)
    elif epoch == 67 and not finetune:
      optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, nesterov=True)
    # length of train_loader = size of dataset / batch size, or # of iterations to equal 1 epoch
    for i, (_, images, masks, normals) in enumerate(train_loader):
      # Send tensors to GPU
      images = images.to(device) # batch_size x 3 x 128 x 128
      masks = masks.to(device)   # batch_size x 128 x 128
      normals = normals.to(device) # batch_size x 3 x 128 x 128

      model.train() # reset model to training mode

      # Forward pass
      outputs = model(images)
      # Rearrange outputs to batch_size x 128 x 128 x 3 to apply masks
      # Output is now _ x 3 (rows of length 3 vectors)
      outputs = outputs.permute(0,2,3,1)[masks,:]
      truths = normals.permute(0,2,3,1)[masks,:]
      loss = get_loss(outputs, truths)
      loss = torch.mean(loss)
      # use backward() to do backprop on loss variable
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i % 50 == 0:
        curr_iter = epoch * len(train_loader) + i
        iterations.append(curr_iter)
        losses.append(loss.item())
        print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'.format
              (epoch+1, num_epochs, i+1, total_steps, loss.item()))
        sys.stdout.flush()
  # Calculate loss over entire training set instead of batch
  final_acc = evaluate(model, device, train_loader)
  print('Final training set accuracy: {}'.format(final_acc))
  print('Making predictions on testing set:')
  make_predictions(model, True, device, test_loader)
  
  # Save the final model
  model.cuda()
  torch.save(model.state_dict(), MODEL_DIR)

# Output predictions as a RGB image (each channel representing a dimension of the surface normal)
def make_predictions(model, test, device, data_loader):
  def inner_func(indexes, images, masks):
    model.eval()
    images = images.to(device)         # batch_size x 3 x 128 x 128
    masks = masks.to(device)           # batch_size x 128 x 128 

    preds = model(images)
    preds = torch.nn.functional.normalize(preds, dim=1)
    # Normalized predictions are between -1 and 1. Get to range 0 to 255
    preds = (preds.permute(0,2,3,1) + 1) / 2 * 255
    folder = TEST_PREDS_DIR if test else TRAIN_PREDS_DIR
    if torch.cuda.is_available():
      preds = preds.cpu().detach().numpy().astype(np.uint8)
    else:
      preds = preds.data.numpy().astype(np.uint8)
    for i in range(0, preds.shape[0]):
      curr_pred = preds[i,:,:,:]
      imwrite('{}/{}.png'.format(folder, indexes[i]), curr_pred)
  if test:
    for indexes, images, masks in data_loader:
      inner_func(indexes, images, masks)
  else:
    for indexes, images, masks, _ in data_loader:
      inner_func(indexes, images, masks)

# Load trained model and make predictions on test set
def from_existing_model():
  model = NormieNet()
  test_params = {'batch_size': 25, 'shuffle': True, 'num_workers': 5}
  test_set = NormalsDataset(is_train = False, transform=transforms.Compose([transforms.ToTensor()]))
  test_loader = data.DataLoader(test_set, **test_params)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  if not torch.cuda.is_available():
    model.load_state_dict(torch.load(MODEL_DIR, map_location=lambda storage, location: storage))
  else:
    model.load_state_dict(torch.load(MODEL_DIR)) 
  make_predictions(model, True, device, test_loader)

if __name__ == '__main__':
  if not os.path.exists(TEST_PREDS_DIR):
    os.makedirs(TEST_PREDS_DIR)
  if not os.path.exists(TRAIN_PREDS_DIR):
    os.makedirs(TRAIN_PREDS_DIR)
  
  train(finetune=False, finetune_epochs=0)
  #train(finetune=True, finetune_epochs=25)

