import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Model Definition

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), #Using ReLU as activation function
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output layer predicts X and Y position
        )

    def forward(self, x):
        return self.net(x)


# Dataset

class RobotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# train()

def train(data_path="hw1_dataset.npz", model_path="mlp_model.pth", epochs=20, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using hardware: {device.type.upper()} ---")

    # Load data
    print("Loading dataset from disk...")
    data = np.load(data_path)
    imgs_before = data['imgs_before']
    actions     = data['action']
    positions   = data['position_after']

    num_samples = imgs_before.shape[0]

    # flaten image and normalise pixels as [0, 1]
    imgs_flat = imgs_before.reshape(num_samples, -1).astype(np.float32) / 255.0

    # One-hot encode the 4 possible push directions
    actions_one_hot = np.zeros((num_samples, 4), dtype=np.float32)
    actions_one_hot[np.arange(num_samples), actions] = 1.0

    positions = positions.astype(np.float32)

    # Concatenate flattened image + action into one input vector
    X = np.hstack((imgs_flat, actions_one_hot))
    y = positions

    # 80/20 split
    split = int(num_samples * 0.8)
    X_train, y_train = X[:split], y[:split]

    input_size = X_train.shape[1]
    print(f"Total input features per sample: {input_size}")
    print(f"Training on {split} samples, testing on {num_samples - split} samples.")

    dataset    = RobotDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(input_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    print("Starting MLP training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss        = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title("MLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig("mlp_loss.png")
    plt.close()
    print("Loss curve saved as 'mlp_loss.png'.")

    print("Training complete! Saving model...")
    torch.save(model.state_dict(), model_path)

    
    print(f"Model saved as '{model_path}'.")


# test()

def test(data_path="hw1_dataset.npz", model_path="mlp_model.pth", num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using hardware: {device.type.upper()} ---")

    # Load data
    data        = np.load(data_path)
    imgs_before = data['imgs_before']
    actions     = data['action']
    positions   = data['position_after']

    num_total = imgs_before.shape[0]

    # Comput input size
    input_size = (imgs_before.shape[1] * imgs_before.shape[2] * imgs_before.shape[3]) + 4
    split = int(num_total * 0.8)
    imgs_test      = imgs_before[split:]
    actions_test   = actions[split:]
    positions_test = positions[split:]

    # Load model
    model = MLP(input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\n--- MLP EVALUATION RESULTS (n={len(imgs_test)} unseen samples) ---")

    all_errors = []
    
    indices = np.random.choice(len(imgs_test), min(num_samples, len(imgs_test)), replace=False)

    for idx in indices:
        img_flat       = imgs_test[idx].reshape(1, -1).astype(np.float32) / 255.0
        action_one_hot = np.zeros((1, 4), dtype=np.float32)
        action_one_hot[0, actions_test[idx]] = 1.0

        X_input  = np.hstack((img_flat, action_one_hot))
        X_tensor = torch.tensor(X_input).to(device)

        with torch.no_grad():
            predicted_pos = model(X_tensor).cpu().numpy()[0]

        real_pos = positions_test[idx]
        error    = np.linalg.norm(predicted_pos - real_pos)
        all_errors.append(error)

        print(
            f"Action: {actions_test[idx]} | "
            f"Pred: [{predicted_pos[0]:.3f}, {predicted_pos[1]:.3f}] | "
            f"Real: [{real_pos[0]:.3f}, {real_pos[1]:.3f}] | "
            f"Error: {error:.4f}"
        )

    # Full test  set MSE
    all_preds, all_reals = [], []
    for i in range(len(imgs_test)):
        img_flat       = imgs_test[i].reshape(1, -1).astype(np.float32) / 255.0
        action_one_hot = np.zeros((1, 4), dtype=np.float32)
        action_one_hot[0, actions_test[i]] = 1.0
        X_input  = np.hstack((img_flat, action_one_hot))
        X_tensor = torch.tensor(X_input).to(device)
        with torch.no_grad():
            pred = model(X_tensor).cpu().numpy()[0]
        all_preds.append(pred)
        all_reals.append(positions_test[i])

    mse = np.mean((np.array(all_preds) - np.array(all_reals)) ** 2)
    print(f"\nTrue Test MSE (200 unseen samples): {mse:.6f}")


# Entry point

if __name__ == "__main__":
    train()
    test()
