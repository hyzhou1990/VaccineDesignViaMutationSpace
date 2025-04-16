import torch
import argparse
import os
from diffusion_model import ProteinDiffusionModel, DiffusionProcess
import numpy as np

def load_model(checkpoint_path, seq_length, hidden_dim, num_layers, num_heads, device):
    model = ProteinDiffusionModel(
        seq_length=seq_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def generate_sequences(
    model,
    diffusion,
    num_sequences,
    seq_length,
    device,
    temperature=1.0
):
    # Generate random noise
    x = torch.randn((num_sequences, seq_length), device=device)
    
    # Generate sequences
    sequences = diffusion.p_sample_loop(model, (num_sequences, seq_length))
    
    # Convert to amino acid sequences
    idx_to_aa = {
        0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C',
        5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
        10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P',
        15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V',
        20: '-'  # Padding token
    }
    
    generated_sequences = []
    for seq in sequences:
        # Convert indices to amino acids
        aa_seq = ''.join([idx_to_aa[idx.item()] for idx in seq])
        # Remove padding
        aa_seq = aa_seq.rstrip('-')
        generated_sequences.append(aa_seq)
    
    return generated_sequences

def main():
    parser = argparse.ArgumentParser(description='Generate protein sequences using diffusion model')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_sequences', type=int, default=10, help='Number of sequences to generate')
    parser.add_argument('--seq_length', type=int, default=1000, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--output_path', type=str, default='generated_sequences.txt', help='Path to save generated sequences')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(
        args.checkpoint_path,
        args.seq_length,
        args.hidden_dim,
        args.num_layers,
        args.num_heads,
        args.device
    )
    
    # Initialize diffusion process
    diffusion = DiffusionProcess()
    
    # Generate sequences
    generated_sequences = generate_sequences(
        model,
        diffusion,
        args.num_sequences,
        args.seq_length,
        args.device,
        args.temperature
    )
    
    # Save sequences
    with open(args.output_path, 'w') as f:
        for i, seq in enumerate(generated_sequences):
            f.write(f'>Generated_sequence_{i+1}\n')
            f.write(f'{seq}\n')
    
    print(f'Generated {len(generated_sequences)} sequences and saved to {args.output_path}')

if __name__ == '__main__':
    main() 