import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from diffusion_model import ProteinDiffusionModel, DiffusionProcess
from tqdm import tqdm
import argparse
import os

class ProteinDataset(Dataset):
    def __init__(self, sequences, max_length=1000):
        self.sequences = sequences
        self.max_length = max_length
        self.aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            '-': 20  # Padding token
        }
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Convert sequence to indices
        indices = [self.aa_to_idx[aa] for aa in seq]
        # Pad sequence
        if len(indices) < self.max_length:
            indices.extend([20] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        return torch.tensor(indices)

def train(
    model,
    diffusion,
    dataloader,
    optimizer,
    device,
    num_epochs=100,
    save_dir='checkpoints'
):
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            # Move batch to device
            batch = batch.to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion.num_timesteps, (batch.size(0),), device=device)
            
            # Sample noise
            noise = torch.randn_like(batch)
            
            # Add noise to data
            noisy_data, true_noise = diffusion.q_sample(batch, t, noise)
            
            # Predict noise
            pred_noise = model(noisy_data, t)
            
            # Calculate loss
            loss = F.mse_loss(pred_noise, true_noise)
            
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        print(f'Epoch {epoch+1} average loss: {epoch_loss / len(dataloader)}')

def main():
    parser = argparse.ArgumentParser(description='Train protein diffusion model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--seq_length', type=int, default=1000, help='Maximum sequence length')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Load data
    with open(args.data_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    # Create dataset and dataloader
    dataset = ProteinDataset(sequences, max_length=args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model and diffusion process
    model = ProteinDiffusionModel(
        seq_length=args.seq_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(args.device)
    
    diffusion = DiffusionProcess()
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Train model
    train(
        model,
        diffusion,
        dataloader,
        optimizer,
        args.device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main() 