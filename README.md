# VaccineDesignViaMutationSpace

A deep learning framework for vaccine design using diffusion models to explore the mutation space of viral proteins.

## Overview

This project implements a diffusion model-based approach for exploring the mutation space of viral proteins, with applications in vaccine design. The model learns to generate and evaluate potential protein sequences that could be used in vaccine development.

## Features

- Diffusion model architecture for protein sequence generation
- Transformer-based backbone for capturing long-range dependencies
- Support for conditional generation based on protein structure
- Efficient training and inference pipelines
- Integration with existing protein structure prediction tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VaccineDesignViaMutationSpace.git
cd VaccineDesignViaMutationSpace
```

2. Create a conda environment:
```bash
conda create -n vaccinedesign python=3.9
conda activate vaccinedesign
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the diffusion model on your protein sequence data:

```bash
python src/train_diffusion.py \
    --data_path data/your_sequences.txt \
    --seq_length 1000 \
    --batch_size 32 \
    --num_epochs 100 \
    --save_dir models/checkpoints
```

### Generating Sequences

To generate new protein sequences using a trained model:

```bash
python src/generate_sequences.py \
    --checkpoint_path models/checkpoints/checkpoint_epoch_100.pt \
    --num_sequences 10 \
    --output_path results/generated_sequences.txt
```

## Project Structure

```
VaccineDesignViaMutationSpace/
├── src/                    # Source code
│   ├── diffusion_model.py  # Core diffusion model implementation
│   ├── train_diffusion.py  # Training script
│   └── generate_sequences.py # Sequence generation script
├── data/                   # Data directory
├── models/                 # Model checkpoints
├── results/               # Generated sequences and analysis
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the EVEscape project
- Inspired by recent advances in protein design using diffusion models 