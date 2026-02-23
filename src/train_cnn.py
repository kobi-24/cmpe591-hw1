import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using hardware: {device.type.upper()} ---")

# Data Loading 
print("Loading dataset from disk...")
data = np.load("hw1_dataset.npz")
imgs_before = data['imgs_before'] 
actions = data['action']          
positions = data['position_after']

num_samples = imgs_before.shape[0]


# Transposing from (Batch, Height, Width, Channels) to (Batch, Channels, Height, Width) for PyTorch
imgs_normalized = imgs_before.astype(np.float32) / 255.0


actions_one_hot = np.zeros((num_samples, 4), dtype=np.float32)
actions_one_hot[np.arange(num_samples), actions] = 1.0


positions = positions.astype(np.float32)

# Create Dataset
class RobotVisionDataset(Dataset):
    def __init__(self, imgs, actions, positions):
        self.imgs = torch.tensor(imgs)
        self.actions = torch.tensor(actions)
        self.positions = torch.tensor(positions)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        return self.imgs[idx], self.actions[idx], self.positions[idx]

dataset = RobotVisionDataset(imgs_normalized, actions_one_hot, positions)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN Architecture
class CNNPredictor(nn.Module):
    def __init__(self):
        super(CNNPredictor, self).__init__()
        
        # Extracts 2D features
        # as input: (3, 128, 128)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # Output:(16, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                # Output: (16,32, 32)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),# Output:(32, 16,16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                 # Output:(32, 8, 8)
        )
        
        # 32 channels * 8 height * 8 width = 2048 flattened features in total
        self.flatten = nn.Flatten()
        
        #  Vision + Action features to predict X,Y
        self.fc = nn.Sequential(
            nn.Linear(2052, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, img, action):
        # Procesing the image
        vision_features = self.cnn(img)
        vision_flat = self.flatten(vision_features)
        
        # Glue the action vector into the vision features
        combined = torch.cat((vision_flat, action), dim=1)
        
        # Predict the coordinates
        output = self.fc(combined)
        return output

model = CNNPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20
print("Starting CNN Training...")
for epoch in range(epochs):
    total_loss = 0
    for batch_img, batch_action, batch_pos in dataloader:
        batch_img, batch_action, batch_pos = batch_img.to(device), batch_action.to(device), batch_pos.to(device)
        
        optimizer.zero_grad()       
        predictions = model(batch_img, batch_action) 
        loss = criterion(predictions, batch_pos) 
        loss.backward()             
        optimizer.step()            
        total_loss += loss.item()
        
    avg_loss = total_loss/len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("CNN Training complete! Saving the model.....")
torch.save(model.state_dict(), "cnn_model.pth")
print("Model successfully saved as 'cnn_model.pth'.")