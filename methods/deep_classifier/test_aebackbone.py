# Imports
import os
from posixpath import split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn Import
from sklearn.metrics import accuracy_score

# PyTorch Imports
import torch
import torchvision

# Utilies Imports
from utilities import MOD_MHSMA, AEBackboneClf



# Directories
data_dir = "data/mod-hushem"
data_splits = "data_splits.csv"



# Open files
splits = pd.read_csv(os.path.join("tiago", data_splits))
# print(splits)


# 0-train, 1-test, 2-val
# Train
# train_split = splits[splits["split"]==0]
# print(train_split)

# Test
# test_split = splits[splits["split"]==1]
# print(test_split)

# Validation
# val_split = splits[splits["split"]==2]
# print(val_split)


# Data
test_data = splits[splits["split"]==1]

# Images
test_images_paths = test_data["img"].values

# Annotators
test_images_annotators = test_data["annotator"].values

# Labels
test_images_labels = test_data["label"].values



# Data
# X Dimensions
CHANNELS = 3
HEIGHT = 128
WIDTH = 128

# Y dimensions
NR_CLASSES = 4



# Model
model = AEBackboneClf(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES)


# Choose GPU
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


# Mean and STD to Normalize
# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]


# Hyper-parameters
BATCH_SIZE = 1


# Load data
# Test
# Transforms
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    # Data Augmentation
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Train Dataset
test_set = MOD_MHSMA(
    data_dir=data_dir,
    images_paths = test_images_paths,
    images_labels = test_images_labels,
    images_annotators = test_images_annotators,
    transform = test_transforms
    )

# Train Dataloader
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

    
# Initialise lists to compute scores
y_annotators = list()
y_test_true = list()
y_test_pred = list()


# Load model weights
# Baseline
# model.load_state_dict(torch.load("tiago/aebackbone.pt", map_location=DEVICE))

# Data Augmentation
model.load_state_dict(torch.load("tiago/daug_aebackbone.pt", map_location=DEVICE))

# Put model in training mode
model.eval()


with torch.no_grad():

    # Iterate through dataloader
    for batch_idx, (images, labels, annotators) in enumerate(test_loader):

        # Move data data anda model to GPU (or not)
        images, labels, annotators = images.to(DEVICE), labels.to(DEVICE), annotators.to(DEVICE)
        model = model.to(DEVICE)

        # Forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images)

        # Concatenate lists
        y_test_true += list(labels.cpu().detach().numpy())
        
        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        s_logits = torch.argmax(s_logits, dim=1)
        y_test_pred += list(s_logits.cpu().detach().numpy())


        # Save annotators
        y_annotators += list(annotators.cpu().detach().numpy())



    # Compute Train Metrics
    test_acc = accuracy_score(y_true=y_test_true, y_pred=y_test_pred)

# Print Statistics
print(f"Test Accuracy: {test_acc}")
# print(y_annotators)
# print(y_test_true) 
# print(y_test_pred)


# Annotators that failed
annotators_failed = list()


# Go through the annotators
for idx, ann in enumerate(y_annotators):
    if y_test_true[idx] != y_test_pred[idx]:
        annotators_failed.append(ann)


# Get the frequencies
unique, counts = np.unique(np.array(annotators_failed), return_counts=True)
print(unique, counts)


# Finish statement
print("Finished.")