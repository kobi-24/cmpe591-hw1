import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using hardware: {device.type.upper()} ---")

# Loading the data
print("Loading dataset from disk...")
data = np.load("hw1_dataset.npz")
imgs_before = data['imgs_before'] 
actions = data['action']          
imgs_after = data['imgs_after'] # future images are predicting

num_samples = imgs_before.shape[0]

# Normalization 
imgs_in_norm = imgs_before.astype(np.float32) / 255.0
imgs_out_norm = imgs_after.astype(np.float32) / 255.0


# One-hot encoding 
actions_one_hot = np.zeros((num_samples, 4), dtype=np.float32)
actions_one_hot[np.arange(num_samples), actions] = 1.0

class ImagePredictionDataset(Dataset):
    def __init__(self, imgs_in, actions, imgs_out):
        self.imgs_in = torch.tensor(imgs_in)
        self.actions = torch.tensor(actions)
        self.imgs_out = torch.tensor(imgs_out)
    def __len__(self):
        return len(self.imgs_in)
    def __getitem__(self, idx):
        return self.imgs_in[idx], self.actions[idx], self.imgs_out[idx]

dataset = ImagePredictionDataset(imgs_in_norm, actions_one_hot, imgs_out_norm)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#  Encoder decoder architecture 
class ImagePredictor(nn.Module):
    def __init__(self):
        super(ImagePredictor, self).__init__()
        
        # Compress 128x128 -> 16x16 with 3 channels encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Output:(32,64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # Output: (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# Output: (128, 16,16)
            nn.ReLU()
        )
        
        # Turn 4 action numbers into a 16x16 grid
        self.action_fc = nn.Linear(4, 16 * 16)
        
        # 3 channel decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(129, 64, kernel_size=4, stride=2, padding=1), # (64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # (3, 128, 128)
            nn.Sigmoid() # between 0 and 1
        )

    def forward(self, img, action):
        # Compressing the image
        enc_features = self.encoder(img) # (Batch, 128, 16, 16)
        
        # Processing the action and reshape to match spatial dimensions
        act_processed = self.action_fc(action) # (Batch, 256)
        act_grid = act_processed.view(-1, 1, 16, 16) # (Batch, 1, 16, 16)
        
        # Concatenating action grid onto image features
        combined = torch.cat((enc_features, act_grid), dim=1) # (Batch, 129, 16, 16)
        

        # Decompressing back into an image
        predicted_img = self.decoder(combined)
        return predicted_img

model = ImagePredictor().to(device)
criterion = nn.MSELoss() # Pixel-to-pixel MSE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20
print("Starting Image-to-Image Training...")
for epoch in range(epochs):
    total_loss = 0
    for batch_img_in, batch_action, batch_img_out in dataloader:
        batch_img_in = batch_img_in.to(device)
        batch_action = batch_action.to(device)
        batch_img_out = batch_img_out.to(device)
        
        optimizer.zero_grad()       
        predictions = model(batch_img_in, batch_action) 
        loss = criterion(predictions, batch_img_out) 
        loss.backward()             
        optimizer.step()            
        total_loss += loss.item()
        
    avg_loss = total_loss/len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("Image Training complete! Saving the model.....")
torch.save(model.state_dict(), "image_predictor_model.pth")
print("Model successfully saved as 'image_predictor_model.pth'.")