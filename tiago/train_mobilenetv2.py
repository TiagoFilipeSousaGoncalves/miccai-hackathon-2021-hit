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
from utilities import map_images_labels_annotators, MOD_MHSMA, MobileNetV2



# Directories
data_dir = "data"
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
train_data = splits[splits["split"]!=1]

# Images
train_images_paths = train_data["img"]

# Annotators
train_images_annotators = train_data["annotator"]

# Labels
train_images_labels = train_data["label"]



# Data
# X Dimensions
CHANNELS = 3
HEIGHT = 128
WIDTH = 128

# Y dimensions
NR_CLASSES = 4



# Model
model = MobileNetV2(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES)


# Choose GPU
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


# Mean and STD to Normalize
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# Hyper-parameters
EPOCHS = 300
LOSS = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 1e-4
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
BATCH_SIZE = 16


# Load data
# Train
# Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    # Data Augmentation
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    # torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Train Dataset
train_set = MOD_MHSMA(
    data_dir=data_dir,
    images_paths = train_images_paths,
    images_labels = train_images_labels,
    iamges_annotators = train_images_annotators,
    transform = train_transforms
    )

# Train Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)



# Train model
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf



# Go through the number of Epochs
for epoch in range(EPOCHS):
    
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Loop
    print(f"Training Phase")
    
    # Initialise lists to compute scores
    y_train_true = list()
    y_train_pred = list()


    # Running train loss
    run_train_loss = 0.0


    # Put model in training mode
    model.train()


    # Iterate through dataloader
    for batch_idx, (images, labels) in enumerate(train_loader):

        # Move data data anda model to GPU (or not)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model = model.to(DEVICE)


        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad()


        # Forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images)
        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()
        
        # Update batch losses
        run_train_loss += (loss.item() * images.size(0))

        # Concatenate lists
        y_train_true += list(labels.cpu().detach().numpy())
        
        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        s_logits = torch.argmax(s_logits, dim=1)
        y_train_pred += list(s_logits.cpu().detach().numpy())
    

    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)

    # Compute Train Metrics
    train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}")


    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss

        # Save checkpoint
        model_path = os.path.join("tiago", "mobilenetv2.pt")
        torch.save(model.state_dict(), model_path)

        print(f"Successfully saved at: {model_path}")


# Finish statement
print("Finished.")