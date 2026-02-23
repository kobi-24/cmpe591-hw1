import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Model Definition

class ImagePredictor(nn.Module):
    def __init__(self):
        super(ImagePredictor, self).__init__()

        # Encoder: compress 128x128 -> 16x16 with 128 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (128, 16, 16)
            nn.ReLU()
        )

        # Project 4-dim action vector into a 16x16 spatial map
        self.action_fc = nn.Linear(4, 16 * 16)

        # Decoder: reconstruct 128x128 image from fused features (128+1 channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(129, 64, kernel_size=4, stride=2, padding=1), # (64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # (3, 128, 128)
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, img, action):
        enc_features  = self.encoder(img)                        # (B, 128, 16, 16)
        act_processed = self.action_fc(action)                   # (B, 256)
        act_grid      = act_processed.view(-1, 1, 16, 16)       # (B, 1, 16, 16)
        combined      = torch.cat((enc_features, act_grid), dim=1) # (B, 129, 16, 16)
        return self.decoder(combined)                            # (B, 3, 128, 128)


# Dataset

class ImagePredictionDataset(Dataset):
    def __init__(self, imgs_in, actions, imgs_out):
        self.imgs_in  = torch.tensor(imgs_in)
        self.actions  = torch.tensor(actions)
        self.imgs_out = torch.tensor(imgs_out)

    def __len__(self):
        return len(self.imgs_in)

    def __getitem__(self, idx):
        return self.imgs_in[idx], self.actions[idx], self.imgs_out[idx]


# train()
def train(data_path="hw1_dataset.npz", model_path="image_predictor_model.pth", epochs=20, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using hardware: {device.type.upper()} ---")

    # Load data
    print("Loading dataset from disk...")
    data        = np.load(data_path)
    imgs_before = data['imgs_before']
    actions     = data['action']
    imgs_after  = data['imgs_after']

    num_samples = imgs_before.shape[0]

    # Normalise to [0, 1] — channel order is NOT changed (already C,H,W from env)
    imgs_in_norm  = imgs_before.astype(np.float32) / 255.0
    imgs_out_norm = imgs_after.astype(np.float32) / 255.0

    # One-hot encode actions
    actions_one_hot = np.zeros((num_samples, 4), dtype=np.float32)
    actions_one_hot[np.arange(num_samples), actions] = 1.0

    # 80/20 split — train on first 800 samples
    split = int(num_samples * 0.8)
    imgs_in_train  = imgs_in_norm[:split]
    actions_train  = actions_one_hot[:split]
    imgs_out_train = imgs_out_norm[:split]
    print(f"Training on {split} samples, testing on {num_samples - split} samples.")

    dataset    = ImagePredictionDataset(imgs_in_train, actions_train, imgs_out_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = ImagePredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    print("Starting Image-to-Image training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_img_in, batch_action, batch_img_out in dataloader:
            batch_img_in  = batch_img_in.to(device)
            batch_action  = batch_action.to(device)
            batch_img_out = batch_img_out.to(device)

            optimizer.zero_grad()
            predictions = model(batch_img_in, batch_action)
            loss        = criterion(predictions, batch_img_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title("Image Predictor Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig("image_loss.png")
    plt.close()
    print("Loss curve saved as 'image_loss.png'.")

    print("Image training complete! Saving model...")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as '{model_path}'.")


# test()

def test(data_path="hw1_dataset.npz", model_path="image_predictor_model.pth", num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using hardware: {device.type.upper()} ---")

    # Load data
    data        = np.load(data_path)
    imgs_before = data['imgs_before']
    actions     = data['action']
    imgs_after  = data['imgs_after']

    num_total = imgs_before.shape[0]

    # Use only the held-out 20% (last 200 samples)
    split = int(num_total * 0.8)
    imgs_test_in   = imgs_before[split:]
    actions_test   = actions[split:]
    imgs_test_out  = imgs_after[split:]

    # Load model
    model = ImagePredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Full test-set MSE
    all_mse = []
    for i in range(len(imgs_test_in)):
        img_in_array  = imgs_test_in[i].astype(np.float32) / 255.0
        img_out_array = imgs_test_out[i].astype(np.float32) / 255.0
        img_in_tensor = torch.tensor(img_in_array).unsqueeze(0).to(device)
        action_one_hot = np.zeros((1, 4), dtype=np.float32)
        action_one_hot[0, actions_test[i]] = 1.0
        action_tensor = torch.tensor(action_one_hot).to(device)
        with torch.no_grad():
            pred = model(img_in_tensor, action_tensor).cpu().squeeze(0).numpy()
        mse = np.mean((pred - img_out_array) ** 2)
        all_mse.append(mse)

    print(f"\nTrue Test MSE (200 unseen samples): {np.mean(all_mse):.6f}")

    # Visual comparison: 3 random unseen samples
    print(f"Generating visual comparisons for {num_samples} samples...")
    indices = np.random.choice(len(imgs_test_in), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3 + 1))
    fig.suptitle("Deliverable 3: Next Frame Prediction", fontsize=16)

    axes[0, 0].set_title("1. Initial State")
    axes[0, 1].set_title("2. Real Future")
    axes[0, 2].set_title("3. AI Predicted Future")

    for i, idx in enumerate(indices):
        img_in_array = imgs_test_in[idx].astype(np.float32) / 255.0
        img_in_tensor = torch.tensor(img_in_array).unsqueeze(0).to(device)

        action_one_hot = np.zeros((1, 4), dtype=np.float32)
        action_one_hot[0, actions_test[idx]] = 1.0
        action_tensor = torch.tensor(action_one_hot).to(device)

        with torch.no_grad():
            predicted_tensor = model(img_in_tensor, action_tensor).cpu().squeeze(0).numpy()

        # Convert (C, H, W) -> (H, W, C) for matplotlib
        predicted_img = np.transpose(predicted_tensor, (1, 2, 0))
        real_in_img   = np.transpose(img_in_array, (1, 2, 0))
        real_out_img  = np.transpose(imgs_test_out[idx].astype(np.float32) / 255.0, (1, 2, 0))

        sample_mse = np.mean((predicted_img - real_out_img) ** 2)

        axes[i, 0].imshow(real_in_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_ylabel(f"Action: {actions_test[idx]}", visible=True)

        axes[i, 1].imshow(real_out_img)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(predicted_img)
        axes[i, 2].axis('off')
        axes[i, 2].set_xlabel(f"MSE: {sample_mse:.4f}", visible=True)

    plt.tight_layout()
    plt.savefig("deliverable3_results.png", dpi=150)
    plt.close()
    print("Results saved as 'deliverable3_results.png'.")


# Entry point

if __name__ == "__main__":
    train()
    test()
