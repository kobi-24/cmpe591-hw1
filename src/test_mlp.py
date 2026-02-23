import torch
import torch.nn as nn
import numpy as np

# Same architecture as training
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

# Loading the data and model with cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading data and model...")
data = np.load("hw1_dataset.npz")
imgs_before = data['imgs_before']
actions = data['action']
positions = data['position_after']

# the input size is 49152 pixels + 4 action directions = 49156
input_size = (imgs_before.shape[1] * imgs_before.shape[2] * imgs_before.shape[3]) + 4

model = MLP(input_size).to(device)
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval() # No learning just testing

# Testingg on 5 random samples
print("\n--- MLP EVALUATION RESULTS ---")
indices = np.random.choice(len(imgs_before), 5, replace=False)

for idx in indices:
    # Preprocessed as in training
    img_flat = imgs_before[idx].reshape(1, -1).astype(np.float32) / 255.0
    action_one_hot = np.zeros((1, 4), dtype=np.float32)
    action_one_hot[0, actions[idx]] = 1.0
    
    X_input = np.hstack((img_flat, action_one_hot))
    X_tensor = torch.tensor(X_input).to(device)
    
    # Predict without calculating gradients for saving memory an time
    with torch.no_grad():
        predicted_pos = model(X_tensor).cpu().numpy()[0]
        
    real_pos = positions[idx]
    
    # Calculateing error distance
    error = np.linalg.norm(predicted_pos - real_pos)
    
    print(f"Action: {actions[idx]} | Pred: [{predicted_pos[0]:.3f}, {predicted_pos[1]:.3f}] | Real: [{real_pos[0]:.3f}, {real_pos[1]:.3f}] | Error: {error:.3f}")