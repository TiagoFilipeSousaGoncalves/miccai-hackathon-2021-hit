from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torchvision
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModHushemDataset(Dataset):

    def __init__(self, root_dir, ids, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        dir_ = ids[idx].split("-")[0]
        id = ids[idx].split("-")[1]
        file = glob.glob(os.path.join(os.getcwd(), "mod-hushem", dir_, "*"+id+"*"))[0]
        img = Image.open(file)
        if self.transform:
            img = self.transform(img)
        
        label = dir_
        sample = {'image': img, 'label': label}
        return sample

class AE(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
          
        # Custom autoencoder 
        # Encoder
        self.encoder = torch.nn.Sequential(
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

        # Decoder
        # The Sigmoid activation function
        # outputs the value between 0 and 1

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 2, 2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 2, 2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, 2, 2),
            torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_shape = (131, 131, 3)    
# Model Initialization
model = AE()
  
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
  
# Using an Adam Optimizer with lr = 0.01
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-2,
                             weight_decay = 1e-8)

epochs = 500
outputs = []
losses = []

ids=[]
root = os.path.join(os.getcwd(), "mod-hushem")
categories = os.listdir(root)

for elem in categories:
    for i in range(len(os.listdir(os.path.join(root, elem)))):
        idx = str(i+1).zfill(2)
        ids.append(elem+'-'+idx)

df = pd.read_csv(os.path.join(os.getcwd(), "file.csv"))

df_train = df[df["split"]==0]
df_test = df[df["split"]==1]
df_val = df[df["split"]==2]

train_ids = [img_id.split("/")[0] + "-" + img_id.split("/")[1] for img_id in df_train["img"]]
test_ids = [img_id.split("/")[0] + "-" + img_id.split("/")[1] for img_id in df_test["img"]]
val_ids = [img_id.split("/")[0] + "-" + img_id.split("/")[1] for img_id in df_val["img"]]

train_ids = train_ids.extend(val_ids)

# Resize images to shape (3, 128, 128)
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
train_dataset = ModHushemDataset(root_dir=root, ids=train_ids, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(epochs):
    for step, batch in enumerate(tqdm(train_dataloader)):
        
      # Reshaping to (batch_size, n_ch, H, W)
      image = batch["image"].reshape(-1,3,128,128)
        
      # Output of Autoencoder
      reconstructed = model(image)

      # Calculating the loss function
      loss = loss_function(reconstructed, image)
        
      # The gradients are set to zero,
      # the the gradient is computed and stored.
      # .step() performs parameter update
      optimizer.zero_grad()
      loss.backward()
      optimizer.step() 
        
      # Storing the losses in a list for plotting
      losses.append(loss.item())
    
    print("Loss: %.5f" % np.mean(np.array(losses)))
    torch.save(model, os.path.join(os.getcwd(), "auto_encoder_train_val.pt"))
    
    if epoch // (epochs -1):

        im = transforms.ToPILImage()(reconstructed[0,:,:,:]).convert("RGB")
        orig = transforms.ToPILImage()(image[0,:,:,:]).convert("RGB")
    
        im.save(os.path.join(os.getcwd(), "example_img_epoch"+str(epoch) +'_reconstructed.png'))
        orig.save(os.path.join(os.getcwd(), "example_img_epoch"+str(epoch) +'_original.png'))

    outputs.append((epochs, image, reconstructed))
  
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
  
# Plotting the last 100 values
plt.plot(losses[-100:])
