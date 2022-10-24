import torch.nn as nn
import torch.nn.functional as F

import utils

class Network(nn.Module):

	# Network definition and layers
	def __init__(self, params):

		super(Network, self).__init__()

		# Retrieving params
		Cin, Hin, Win=params["shape_in"]
		init_f = params["initial_filters"]
		num_fc1 = params["num_fc1"]
		num_classes = params["num_classes"]
		self.dropout_rate = params["dropout_rate"]
		
		# Convolution layers
		self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
		h, w = utils.findConv2dOutShape(Hin, Win, self.conv1)
		self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
		h, w = utils.findConv2dOutShape(h, w, self.conv2)
		self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
		h, w = utils.findConv2dOutShape(h, w, self.conv3)
		self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
		h, w = utils.findConv2dOutShape(h, w, self.conv4)
		
		# Compute the flatten size
		self.num_flatten = h * w * 8 * init_f
		self.fc1 = nn.Linear(self.num_flatten, num_fc1)
		self.fc2 = nn.Linear(num_fc1, num_classes)

	def forward(self,X):
		
		# Convolution & Pool Layers
		X = F.relu(self.conv1(X))
		X = F.max_pool2d(X, 2, 2)
		
		X = F.relu(self.conv2(X))
		X = F.max_pool2d(X, 2, 2)
		
		X = F.relu(self.conv3(X))
		X = F.max_pool2d(X, 2, 2)
		
		X = F.relu(self.conv4(X))
		X = F.max_pool2d(X, 2, 2)

		X = X.view(-1, self.num_flatten)
		
		X = F.relu(self.fc1(X))
		X = F.dropout(X, self.dropout_rate)
		X = self.fc2(X)
		return F.log_softmax(X, dim=1)