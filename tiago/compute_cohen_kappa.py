# Imports
import os
import numpy as np
import pandas as pd
import PIL

# Sklearn Import
from sklearn.metrics import cohen_kappa_score

# PyTorch Imports
import torch
import torchvision

# Utilies Imports
from utilities import AEBackboneClf



# Directories
data_dir = "data/mod-hushem"
data_splits = "data_splits.csv"


# Global variables
# X Dimensions
CHANNELS = 3
HEIGHT = 128
WIDTH = 128

# Y dimensions
NR_CLASSES = 4
NR_ANNOTATORS = 3

# Choose GPU
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"



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
# test_images_paths = test_data["img"].values

# Annotators
# test_images_annotators = test_data["annotator"].values

# Labels
# test_images_labels = test_data["label"].values




# Load data
# Test
# Transforms
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
])



# Load Model
model = AEBackboneClf(channels=CHANNELS, height=HEIGHT, width=WIDTH, nr_classes=NR_CLASSES)


# Load model weights
# Baseline
# model.load_state_dict(torch.load("tiago/aebackbone.pt", map_location=DEVICE))

# Data Augmentation
model.load_state_dict(torch.load("tiago/daug_aebackbone.pt", map_location=DEVICE))

# Put model in training mode
model.eval()


# Compute Cohen's Kappa per Annotator
with torch.no_grad():

    # Compute Annotator Consistency
    print("Computing annotator consistency...")

    # Go through all the annotators
    for i in range(1, NR_ANNOTATORS+1):

        # Get the annotator subset
        ann_test_df = test_data[test_data['annotator'] == i]
        
        # Create lists for the annotator and model predictions
        preds_annotator = []
        preds_model = [] 

        # Go through this dataframe
        for _, row in ann_test_df.iterrows():

            # Open image file
            img = PIL.Image.open(os.path.join(data_dir, row['img']) + '.BMP')
            img = test_transforms(img)
            img = img.to(DEVICE)
            img = img[None, :, :, :]


            # Get predictions
            logits = model(img)
            
            # Save predictions of the model
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)
            s_logits = torch.argmax(s_logits, dim=1)
            preds_model += list(s_logits.cpu().detach().numpy())

            # Save predictions of the annotator
            preds_annotator.append(row['label'])

            
        # Compute Cohen Kappa Score
        ck_class = cohen_kappa_score(preds_annotator, preds_model, labels=[0, 1, 2, 3])


        # Print results
        print(f"Annotator {i} Cohen's kappa: {np.round(ck_class, 4)}")