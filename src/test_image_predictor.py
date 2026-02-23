import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Rebuilding the exact architecture
class ImagePredictor(nn.Module):
    def __init__(self):
        super(ImagePredictor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.action_fc = nn.Linear(4, 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(129, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, img, action):
        enc_features = self.encoder(img)
        act_processed = self.action_fc(action)
        act_grid = act_processed.view(-1, 1, 16, 16)
        combined = torch.cat((enc_features, act_grid), dim=1)
        return self.decoder(combined)

# Setting up and loading the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading data and model...")
data = np.load("hw1_dataset.npz")
imgs_before = data['imgs_before']
actions = data['action']
imgs_after = data['imgs_after']

model = ImagePredictor().to(device)
model.load_state_dict(torch.load("image_predictor_model.pth"))
model.eval()

# Visualizing 3 random samples
print("Generating visual comparisons...")
indices = np.random.choice(len(imgs_before), 3, replace=False)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle("Deliverable 3: Next Frame Prediction", fontsize=16)

# Column Headers
axes[0, 0].set_title("1. Initial State")
axes[0, 1].set_title("2. Real Future")
axes[0, 2].set_title("3. AI Predicted future")

for i, idx in enumerate(indices):
    # Prepareing the input image
    img_in_array = imgs_before[idx].astype(np.float32) / 255.0
    img_in_tensor = torch.tensor(img_in_array).unsqueeze(0).to(device)
    
    # Prepare action
    action_one_hot = np.zeros((1, 4), dtype=np.float32)
    action_one_hot[0, actions[idx]] = 1.0
    action_tensor = torch.tensor(action_one_hot).to(device)
    
    # Predicting
    with torch.no_grad():
        predicted_tensor = model(img_in_tensor, action_tensor).cpu().squeeze(0).numpy()
    
    #  (C, H, W) to (H, W, C) for matplotlib pklotting
    predicted_img = np.transpose(predicted_tensor, (1, 2, 0))
    real_in_img = np.transpose(img_in_array, (1, 2, 0))
    real_out_img = np.transpose(imgs_after[idx].astype(np.float32) / 255.0, (1, 2, 0))
    
    # Ploting part
    axes[i, 0].imshow(real_in_img)
    axes[i, 0].axis('off')
    axes[i, 0].set_ylabel(f"Action: {actions[idx]}", visible=True)
    
    axes[i, 1].imshow(real_out_img)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(predicted_img)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()