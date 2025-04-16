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
    # 生成随机噪声
    x = torch.randn((num_sequences, seq_length), device=device)
    
    # 生成序列
    with torch.no_grad():
        sequences = diffusion.p_sample_loop(model, (num_sequences, seq_length))
    
    # 将连续值转换为氨基酸索引
    # 我们将连续空间映射到20种氨基酸，简单起见，使用均匀区间
    bins = np.linspace(-2, 2, 21)  # 20个氨基酸+1个边界
    
    idx_to_aa = {
        0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C',
        5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
        10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P',
        15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V',
        20: '-'  # 填充符号
    }
    
    generated_sequences = []
    for seq in sequences:
        # 将连续值数字化为氨基酸索引
        seq_np = seq.cpu().numpy()
        idx_seq = np.digitize(seq_np, bins) - 1  # -1 是因为digitize给出的是右边界的索引
        idx_seq = np.clip(idx_seq, 0, 20)  # 确保所有索引在有效范围内
        
        # 转换索引为氨基酸
        aa_seq = ''.join([idx_to_aa[idx] for idx in idx_seq])
        # 移除填充
        aa_seq = aa_seq.rstrip('-')
        generated_sequences.append(aa_seq)
    
    return generated_sequences

def main():
    parser = argparse.ArgumentParser(description='使用扩散模型生成蛋白质序列')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--num_sequences', type=int, default=10, help='生成序列数量')
    parser.add_argument('--seq_length', type=int, default=1000, help='序列长度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏维度')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--output_path', type=str, default='generated_sequences.txt', help='保存生成序列的路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='使用设备')
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(
        args.checkpoint_path,
        args.seq_length,
        args.hidden_dim,
        args.num_layers,
        args.num_heads,
        args.device
    )
    
    # 初始化扩散过程
    diffusion = DiffusionProcess(device=args.device)
    
    # 生成序列
    generated_sequences = generate_sequences(
        model,
        diffusion,
        args.num_sequences,
        args.seq_length,
        args.device,
        args.temperature
    )
    
    # 保存序列
    with open(args.output_path, 'w') as f:
        for i, seq in enumerate(generated_sequences):
            f.write(f'>Generated_sequence_{i+1}\n')
            f.write(f'{seq}\n')
    
    print(f'生成了 {len(generated_sequences)} 个序列并保存到 {args.output_path}')

if __name__ == '__main__':
    main() 