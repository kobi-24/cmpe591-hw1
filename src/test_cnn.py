import torch
import torch.nn as nn
import numpy as np

# Rebuilding  CNN Architecture
class CNNPredictor(nn.Module):
    def __init__(self):
        super(CNNPredictor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(2052, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, img, action):
        vision_features = self.cnn(img)
        vision_flat = self.flatten(vision_features)
        combined = torch.cat((vision_flat, action), dim=1)
        return self.fc(combined)

# Setting up and Loading the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading data and model...")
data = np.load("hw1_dataset.npz")
imgs_before = data['imgs_before']
actions = data['action']
positions = data['position_after']

model = CNNPredictor().to(device)
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval() 

# Testing on 5 random samples
print("\n--- CNN EVALUATION RESULTS ----")
indices = np.random.choice(len(imgs_before), 5, replace=False)

for idx in indices:
    

    img_array = imgs_before[idx].astype(np.float32) / 255.0

    img_tensor = torch.tensor(img_array).unsqueeze(0).to(device) 
    
    action_one_hot = np.zeros((1, 4), dtype=np.float32)
    action_one_hot[0, actions[idx]] = 1.0
    action_tensor = torch.tensor(action_one_hot).to(device)
    
    with torch.no_grad():
        predicted_pos = model(img_tensor, action_tensor).cpu().numpy()[0]
        
    real_pos = positions[idx]
    
    error = np.linalg.norm(predicted_pos - real_pos)
    
    print(f"Action: {actions[idx]} | Pred: [{predicted_pos[0]:.3f}, {predicted_pos[1]:.3f}] | Real: [{real_pos[0]:.3f}, {real_pos[1]:.3f}] | Error: {error:.3f}")