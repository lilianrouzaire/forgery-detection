from IPython.display import Image, display
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, models, transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import utils
from Network import Network


data_transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize((46, 46))])

data_dir = "input/shuffled"
img_dataset = utils.pytorch_data(data_dir, data_transformer, "train")

"""
# Identify images that are black and white
#utils.grayscale_images('input/shuffled/train')
for img, label, name in img_dataset:
	if(img.size(0) != 3):
		print(name)
# end: clean dataset
"""

split_ratio = 0.8

len_img = len(img_dataset)
len_train = int(split_ratio * len_img)
len_val = len_img - len_train

# Split Pytorch tensor
train_ts, val_ts = random_split(img_dataset, [len_train, len_val])

print("train dataset size:", len(train_ts))
print("validation dataset size:", len(val_ts))

### Dataloaders
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ts, batch_size=32, shuffle=False)


### Model

params_model = {
	"shape_in": (3,46,46),
	"initial_filters": 8,
	"num_fc1": 100,
	"dropout_rate": 0.25,
	"num_classes": 2 # real or fake
}

model = Network(params_model)

### If GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Learning rate
lr = 3e-4

# Loss function (using binary cross-entropy)
loss_fn = nn.NLLLoss(reduction='sum')

# Optimizer (Adam looks like a good idea)
optimizer = optim.Adam(model.parameters(), lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=True) # Learning rate adapter

# Training parameters
params_train = {
	"train": train_dl,
	"val": val_dl,
	"epochs": 30,
	"optimiser": optim.Adam(model.parameters(), lr),
	"lr_change": lr_scheduler,
	"loss_fn": loss_fn,
	"weight_path": "weights.pt", # File for local storage of weights
	"check": False,
	"device": device
}

### Training + validation
model, loss_logs, metric_logs = utils.train_val(model, params_train)

# Train-Validation Progress
epochs=params_train["epochs"]

print(metric_logs)

"""
fig = make_subplots(rows=1, cols=2,subplot_titles=['lost_hist','metric_hist'])
fig.add_trace(go.Scatter(x=[*range(1,epochs+1)], y=loss_hist["train"],name='loss_hist["train"]'),row=1, col=1)
fig.add_trace(go.Scatter(x=[*range(1,epochs+1)], y=loss_hist["val"],name='loss_hist["val"]'),row=1, col=1)
fig.add_trace(go.Scatter(x=[*range(1,epochs+1)], y=metric_hist["train"],name='metric_hist["train"]'),row=1, col=2)
fig.add_trace(go.Scatter(x=[*range(1,epochs+1)], y=metric_hist["val"],name='metric_hist["val"]'),row=1, col=2)
fig.update_layout(template='plotly_white');fig.update_layout(margin={"r":0,"t":60,"l":0,"b":0},height=300)
fig.show()
"""