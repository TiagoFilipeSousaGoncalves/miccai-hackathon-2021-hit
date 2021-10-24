# Imports
import numpy as np
import os
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import Dataset


# Data Directory
data_dir = "data/mod-hushem"

# Data Labels
data_labels = ["amorphous", "normal", "pyriform", "tapered"]


# Helper Function: Get images and labels from directory files
def map_images_labels_annotators(data_dir, data_labels):
    
    # Images Paths
    images_paths = list()

    # Labels
    images_labels = list()

    # Annotators
    images_annotators = list()



    # Go throught the folders paths
    for idx, label_path in enumerate(data_labels):
        
        # Get the label dir
        label_dir = os.path.join(data_dir, label_path)

        
        # Get dir files
        dir_files = os.listdir(label_dir)
        dir_files = [i for i in dir_files if not i.startswith('.')]
        dir_files.sort()


        # Go through these files and put their informations into our three lists
        for image_file in dir_files:
            
            # Image Path
            image_path = os.path.join(label_path, image_file)

            # Label
            image_label = idx

            # Annotator
            image_annotator = image_file.split('.')[0]
            image_annotator = image_annotator.split('_')[2]
            image_annotator = image_annotator[3]
            image_annotator = int(image_annotator)


            # Fill lists
            images_paths.append(image_path)
            images_labels.append(image_label)
            images_annotators.append(image_annotator)


    return images_paths, images_labels, images_annotators




# Create a Dataset Class
class MOD_MHSMA(Dataset):
    def __init__(self, data_dir, images_paths, images_labels, images_annotators, transform=None):
    
        """
        Args:
            images_paths
            images_labels
            images_annotators
            transform (callable, optional): Optional transform to be applied on a sample.
        """


        # Init variables
        self.data_dir = data_dir
        self.images_paths, self.images_labels, self.images_annotators = images_paths, images_labels, images_annotators
        self.transform = transform


        return 


    # Method: __len__
    def __len__(self):
        return len(self.images_paths)



    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        # Get images
        img_path = self.images_paths[idx] + ".BMP"
        
        # Open image with PIL
        image = Image.open(os.path.join(self.data_dir, img_path))

        # Get labels
        label = self.images_labels[idx]

        # Get annotator
        annotator = self.images_annotators[idx]


        # Apply transformation
        if self.transform:
            image = self.transform(image)


        return image, label, annotator



# Model: MobileNetV2
class MobileNetV2(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes):
        super(MobileNetV2, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes


        # Init modules
        # Backbone to extract features
        self.mobilenetv2 = torch.hub.load('pytorch/vision:v0.8.0', 'mobilenet_v2', pretrained=True).features

        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.mobilenetv2(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Dropout
        self.dropout = torch.nn.Dropout(p=0.2)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)

        # Sigmoid Activation Layer
        self.fc_sigmoid = torch.nn.Sigmoid()


        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.mobilenetv2(inputs)

        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))

        # Dropout
        outputs = self.dropout(features)
        
        # FC1-Layer
        outputs = self.fc1(outputs)

        # Activation layer
        outputs = self.fc_sigmoid(outputs)


        return outputs



# Model: AEBackbone w/ Classifier
class AEBackboneClf(torch.nn.Module):
    
    def __init__(self, channels, height, width, nr_classes):
        super(AEBackboneClf, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes

          
        # Custom autoencoder 
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
        )

        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.features(_in_features)


        # Apply GlobalAveragePooling
        self.globalaveragepool = torch.nn.AvgPool2d(kernel_size=_in_features.size(2))


        # FC-Layers
        
        _in_features = self.globalaveragepool(_in_features)
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # FC1 Layer
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=256)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(p=0.5)

        # FC2 Layer
        self.fc2 = torch.nn.Linear(in_features=256, out_features=4)


        return

  
    def forward(self, inputs):
        
        # Features
        features = self.features(inputs)

        # Apply Global Average Pooling
        outputs = self.globalaveragepool(features)

        # FC1-Layer
        outputs = self.fc1(outputs)
        outputs = self.relu1(outputs)
        outputs = self.drop1(outputs)
        
        # FC2-Layer
        outputs = self.fc2(outputs)


        return outputs