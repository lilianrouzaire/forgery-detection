import os
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from IPython.core.pylabtools import figsize
import pandas as pd
from torch.utils.data import Dataset
import PIL.Image
import copy

torch.manual_seed(0) # fix random seed

def grayscale_images(dir_path):
	directory = os.fsencode(dir_path)

	for file in os.listdir(directory):
		filename = os.fsdecode(file)
		img = cv2.imread(dir_path + '/' + filename)
		if(img.shape[2] != 3):
			print(img.shape[2])

def build_labels_array():
	fake_directory = os.fsencode('input/byclass/real')
	real_directory = os.fsencode('input/byclass/fake')

	labels = []
	
	for file in os.listdir(real_directory):
		filename = os.fsdecode(file)
		file_id = os.path.splitext(filename)[0]
		labels.append([file_id, 1])

	for file in os.listdir(fake_directory):
		filename = os.fsdecode(file)
		file_id = os.path.splitext(filename)[0]
		labels.append([file_id, 0])

	df = pd.DataFrame(labels, columns = ['id', 'label'])

	df.to_csv('input/shuffled/train_labels.csv', index=False)

	return df

# Compute the output size of a CNN
def findConv2dOutShape(H_in, W_in, conv, pool = 2):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    
    H_out = np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    W_out = np.floor((W_in+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)
    
    if pool:
        H_out /= pool
        W_out /= pool
        
    return int(H_out),int(W_out)

class pytorch_data(Dataset):
	
	def __init__(self, data_dir, transform, data_type="train"):      
	
		# Get Image File Names
		cdm_data = os.path.join(data_dir,data_type)  # directory of files
		
		file_names = os.listdir(cdm_data) # get list of images in that directory  
		idx_choose = np.random.choice(np.arange(len(file_names)), 
									  len(file_names),
									  replace=False).tolist()
		file_names_sample = [file_names[x] for x in idx_choose]
		self.full_filenames = [os.path.join(cdm_data, f) for f in file_names_sample]   # get the full path to images
		
		# Get Labels
		labels_data = os.path.join(data_dir,"train_labels.csv") 
		labels_df = pd.read_csv(labels_data)
		labels_df.set_index("id", inplace=True) # set data frame index to id
		self.labels = [labels_df.loc[os.path.splitext(filename)[0]].values[0] for filename in file_names_sample]  # obtained labels from df
		self.transform = transform

	def __len__(self):
		return len(self.full_filenames) # size of dataset
	  
	def __getitem__(self, idx):
		# open image, apply transforms and return with label
		image = PIL.Image.open(self.full_filenames[idx])  # Open Image with PIL
		image = self.transform(image) # Apply Specific Transformation to Image
		#return image, self.labels[idx], self.full_filenames[idx]
		return image, self.labels[idx]

	def __remove__(self, idx):
		pass

# Function to get the learning rate
def get_lr(opt):
	for param_group in opt.param_groups:
		return param_group['lr']

# Computing the loss value per batch
def loss_batch(loss_fn, output, target, optim=None):
	
	loss = loss_fn(output, target) # Retrieve loss value
	pred = output.argmax(dim=1, keepdim=True) # Get Output Class
	metric_b = pred.eq(target.view_as(pred)).sum().item() # Performance metric over the batch (classification accuracy)
	
	if optim is not None:
		optim.zero_grad()
		loss.backward()
		optim.step()

	return loss.item(), metric_b

# Computing the loss value per epoch (using the function loss_batch())
def loss_epoch(model, loss_fn, dataset_dl, device, check=False, optim=None):

	e_loss = 0.0
	e_metric = 0.0

	for x, y in dataset_dl:

		# If GPU available
		#x = x.to(device)
		#y = y.to(device)

		output = model(x) # Predict

		loss_b, metric_b = loss_batch(loss_fn, output, y, optim) # Compute batch loss

		e_loss += loss_b

		if metric_b is not None:
			e_metric += metric_b

		if check is True:
			break;

	data_length = len(dataset_dl.dataset)

	loss = e_loss / data_length
	metric = e_metric / data_length

	return loss, metric

# Training function
def train_val(model, params, verbose=False):

	# Retrieve the parameters from the array
	epochs = params["epochs"]
	loss_fn = params["loss_fn"]
	optim = params["optimiser"]
	train_dl = params["train"]
	val_dl = params["val"]
	check = params["check"]
	lr_scheduler = params["lr_change"]
	weight_path = params["weight_path"]
	device = params["device"]

	loss_logs = {'train': [], 'val': []}
	metric_logs = {'train': [], 'val': []}

	best_model_weights = copy.deepcopy(model.state_dict()) # a deep copy of weights for the best performing model
	best_loss = float('inf') # initialize best loss to a large value
	
	# main loop
	for epoch in range(epochs):

		# Get the Learning Rate
		current_lr  = get_lr(optim)
		if(verbose):
			print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))
		
		# Train the Model on the training set
		model.train()
		train_loss, train_metric = loss_epoch(model, loss_fn, train_dl, device, check, optim)

		# Collect loss and metric for training dataset
		loss_logs['train'].append(train_loss)
		metric_logs['train'].append(train_metric)
		
		# Evaluate model on validation set
		model.eval()
		with torch.no_grad():
			val_loss, val_metric = loss_epoch(model, loss_fn, val_dl, check)
		
		# Store best model
		if val_loss < best_loss:
			best_loss = val_loss
			best_model_weights = copy.deepcopy(model.state_dict())
			
			# Store weights into a local file
			torch.save(model.state_dict(), weight_path)
			if(verbose):
				print("Copied best model weights!")

		# Collect loss and metric for validation dataset
		loss_logs['val'].append(val_loss)
		metric_logs['val'].append(val_metric)
		
		# Learning rate scheduler
		lr_scheduler.step(val_loss)
		if current_lr != get_lr(optim):
			if(verbose):
				print("Loading best model weights!")
			model.load_state_dict(best_model_weights) 

		if(verbose):
			print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}")
			print("-"*10) 

	# Load best model weights
	model.load_state_dict(best_model_weights)
		
	return model, loss_logs, metric_logs