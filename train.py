import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import mlflow
import mlflow.pytorch

# 1. Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# CONFIGURATION
# ==========================================
EPOCHS = 5
BATCH_SIZE = 512
LEARNING_RATE = 0.0002
LATENT_DIM = 100
# ==========================================

# --- DATA GENERATION (Simulating DVC Pull or Data Prep) ---
print("Downloading/Preparing dataset...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True)
num_samples = 5000 # Reduced for faster CI/CD testing
images = mnist_train.data.numpy()[:num_samples].reshape(num_samples, 784)
labels = mnist_train.targets.numpy()[:num_samples]

df = pd.DataFrame(images)
df.insert(0, 'label', labels)
df.to_csv('data.csv', index=False)

# --- DATASET & MODEL DEFINITIONS ---
class CSVImageDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.images = self.data.drop('label', axis=1).values if 'label' in self.data.columns else self.data.values
        self.images = (self.images / 127.5) - 1.0
        self.images = torch.tensor(self.images, dtype=torch.float32)

    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx]

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim), nn.Tanh() 
        )
    def forward(self, x): return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.model(x)

# --- MLFLOW EXPERIMENT TRACKING ---
mlflow.set_experiment("Assignment5_Pipeline")

with mlflow.start_run() as run:
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    
    device = torch.device("cpu") # Force CPU for GitHub Actions
    dataset = CSVImageDataset('data.csv')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    avg_d_acc = 0.0
    for epoch in range(EPOCHS):
        epoch_d_acc = 0.0
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels, fake_labels = torch.ones(b_size, 1).to(device), torch.zeros(b_size, 1).to(device)

            outputs_real = discriminator(real_imgs)
            d_loss_real = criterion(outputs_real, real_labels)
            acc_real = (outputs_real >= 0.5).float().mean().item()

            z = torch.randn(b_size, LATENT_DIM).to(device)
            fake_imgs = generator(z)
            outputs_fake = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            acc_fake = (outputs_fake < 0.5).float().mean().item()

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_imgs)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            epoch_d_acc += (acc_real + acc_fake) / 2.0

        avg_d_acc = epoch_d_acc / len(dataloader)
        mlflow.log_metric("accuracy", avg_d_acc, step=epoch)

    print("Training complete. Saving model...")
    mlflow.pytorch.log_model(generator, "generator_model")
    
    # --- EXPORT RUN ID FOR PIPELINE ---
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)
    print(f"Run completed! Run ID {run_id} saved to model_info.txt")
