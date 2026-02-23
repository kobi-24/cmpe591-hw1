import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Model Definition

class CNNPredictor(nn.Module):
    def __init__(self):
        super(CNNPredictor, self).__init__()

        # Convolutional feature extractor
        # Input: (3, 128, 128)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  #(16, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # (16,32, 32)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (32, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                   #(32, 8,8)
        )

        # 32 * 8 * 8 = 2048 flatened features + 4 action fetures = 2052
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(2052, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Predict X and Y position
        )

    def forward(self, img, action):
        vision_features = self.cnn(img)
        vision_flat     = self.flatten(vision_features)
        combined        = torch.cat((vision_flat, action), dim=1)
        return self.fc(combined)


# Dataset

class RobotVisionDataset(Dataset):
    def __init__(self, imgs, actions, positions):
        self.imgs      = torch.tensor(imgs)
        self.actions   = torch.tensor(actions)
        self.positions = torch.tensor(positions)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.actions[idx], self.positions[idx]


# train()

def train(data_path="hw1_dataset.npz", model_path="cnn_model.pth", epochs=20, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using hardware: {device.type.upper()} ---")

    # Load data
    print("Loading dataset from disk...")
    data        = np.load(data_path)
    imgs_before = data['imgs_before']
    actions     = data['action']
    positions   = data['position_after']

    num_samples = imgs_before.shape[0]

    # Normalise to [0, 1] — channel order is NOT changed (already C,H,W from env)
    imgs_normalized = imgs_before.astype(np.float32) / 255.0

    # One-hot encode actions
    actions_one_hot = np.zeros((num_samples, 4), dtype=np.float32)
    actions_one_hot[np.arange(num_samples), actions] = 1.0

    positions = positions.astype(np.float32)

    # 80/20 split again
    split = int(num_samples * 0.8)
    imgs_train    = imgs_normalized[:split]
    actions_train = actions_one_hot[:split]
    pos_train     = positions[:split]
    print(f"Training on {split} samples, testing on {num_samples - split} samples.")

    dataset    = RobotVisionDataset(imgs_train, actions_train, pos_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = CNNPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    print("Starting CNN training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_img, batch_action, batch_pos in dataloader:
            batch_img, batch_action, batch_pos = (
                batch_img.to(device),
                batch_action.to(device),
                batch_pos.to(device),
            )

            optimizer.zero_grad()
            predictions = model(batch_img, batch_action)
            loss        = criterion(predictions, batch_pos)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title("CNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig("cnn_loss.png")
    plt.close()
    print("Loss curve saved as 'cnn_loss.png'.")

    print("CNN training complete! Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'.")


# test()

def test(data_path="hw1_dataset.npz", model_path="cnn_model.pth", num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using hardware: {device.type.upper()} ---")

    # Load data
    data        = np.load(data_path)
    imgs_before = data['imgs_before']
    actions     = data['action']
    positions   = data['position_after']

    num_total = imgs_before.shape[0]

   
    split = int(num_total * 0.8)
    imgs_test      = imgs_before[split:]
    actions_test   = actions[split:]
    positions_test = positions[split:]

    # Load model
    model = CNNPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\n--- CNN EVALUATION RESULTS (n={len(imgs_test)} unseen samples) ---")
    indices = np.random.choice(len(imgs_test), min(num_samples, len(imgs_test)), replace=False)

    for idx in indices:
        img_array = imgs_test[idx].astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_array).unsqueeze(0).to(device)

        action_one_hot = np.zeros((1, 4), dtype=np.float32)
        action_one_hot[0, actions_test[idx]] = 1.0
        action_tensor = torch.tensor(action_one_hot).to(device)

        with torch.no_grad():
            predicted_pos = model(img_tensor, action_tensor).cpu().numpy()[0]

        real_pos = positions_test[idx]
        error    = np.linalg.norm(predicted_pos - real_pos)

        print(
            f"Action: {actions_test[idx]} | "
            f"Pred: [{predicted_pos[0]:.3f}, {predicted_pos[1]:.3f}] | "
            f"Real: [{real_pos[0]:.3f}, {real_pos[1]:.3f}] | "
            f"Error: {error:.4f}"
        )

    # Full test-set MSE
    all_preds, all_reals = [], []
    for i in range(len(imgs_test)):
        img_array = imgs_test[i].astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_array).unsqueeze(0).to(device)
        action_one_hot = np.zeros((1, 4), dtype=np.float32)
        action_one_hot[0, actions_test[i]] = 1.0
        action_tensor = torch.tensor(action_one_hot).to(device)
        with torch.no_grad():
            pred = model(img_tensor, action_tensor).cpu().numpy()[0]
        all_preds.append(pred)
        all_reals.append(positions_test[i])

    mse = np.mean((np.array(all_preds) - np.array(all_reals)) ** 2)
    print(f"\nTrue Test MSE (200 unseen samples): {mse:.6f}")


# Entry point

if __name__ == "__main__":
    train()
    test()
