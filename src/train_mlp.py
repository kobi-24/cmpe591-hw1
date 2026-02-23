import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Using cuda for acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using hardware: {device.type.upper()} ---")


# Data Loading
print("-Loading the Dataset from the disk-")
data = np.load("hw1_dataset.npz")
imgs_before = data['imgs_before'] 
actions = data['action']          
positions = data['position_after']

num_samples = imgs_before.shape[0]

# Noramlized pixels to be between 0 and 1
imgs_flat = imgs_before.reshape(num_samples, -1).astype(np.float32) / 255.0

# 4 possible directions 
actions_one_hot = np.zeros((num_samples, 4), dtype=np.float32)
actions_one_hot[np.arange(num_samples), actions] = 1.0

positions = positions.astype(np.float32)

# Combining the flatend image and the action in one input vector
X = np.hstack((imgs_flat, actions_one_hot))
y = positions

# Creating the PyTorch Dataset
class RobotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = RobotDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# MLP Part 
input_size = X.shape[1]
print(f"Total Input Features per sample: {input_size}")


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),#Using ReLU activation function
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Output layer: predicts X and Y
        )
    def forward(self, x):
        return self.net(x)

# Sending  the model in to the GPU
model = MLP(input_size).to(device)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop
epochs = 20
print("Starting Training...")
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        # Sending the batches into the GPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()       # Clearing the  old gradients
        predictions = model(batch_X) # Forward pass
        loss = criterion(predictions, batch_y) # Calculate error
        loss.backward()             # Back propogation
        optimizer.step()            # Update weights
        total_loss += loss.item()
        
    avg_loss = total_loss/len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("Training complete! Saving the model.....")
torch.save(model.state_dict(), "mlp_model.pth")
print("Model succesfully saved as 'mlp_model.pth'.")