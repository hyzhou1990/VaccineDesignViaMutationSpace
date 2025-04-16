import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from diffusion_model import ProteinDiffusionModel, DiffusionProcess
from train_diffusion import ProteinDataset, train
from generate_sequences import generate_sequences, load_model
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_parser():
    parser = argparse.ArgumentParser(description='Debug and analyze protein diffusion model')
    parser.add_argument('--data_path', type=str, default='data/test_sequences.txt', help='Path to test data')
    parser.add_argument('--seq_length', type=int, default=500, help='Maximum sequence length')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'analyze'], default='train', help='Mode')
    parser.add_argument('--checkpoint_path', type=str, default='models/debug_checkpoint.pt', help='Path to checkpoint')
    parser.add_argument('--output_path', type=str, default='results/debug_generated.txt', help='Path to save outputs')
    parser.add_argument('--num_sequences', type=int, default=5, help='Number of sequences to generate')
    
    return parser

def analyze_sequences(sequences, original_sequences=None):
    """分析生成的序列，计算各种统计数据"""
    results = {}
    
    # 氨基酸频率统计
    aa_counts = {}
    for seq in sequences:
        for aa in seq:
            if aa == '-':  # 跳过填充符号
                continue
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    total_aa = sum(aa_counts.values())
    aa_freqs = {aa: count / total_aa for aa, count in aa_counts.items()}
    results['aa_frequencies'] = aa_freqs
    
    # 序列长度统计
    seq_lengths = [len(seq.rstrip('-')) for seq in sequences]
    results['seq_length_mean'] = np.mean(seq_lengths)
    results['seq_length_std'] = np.std(seq_lengths)
    
    # 计算序列特性
    properties = {
        'molecular_weight': [],
        'aromaticity': [],
        'instability_index': [],
        'isoelectric_point': [],
        'gravy': []  # Grand average of hydropathy
    }
    
    for seq in sequences:
        seq = seq.rstrip('-')  # 移除填充字符
        try:
            analysis = ProteinAnalysis(seq)
            properties['molecular_weight'].append(analysis.molecular_weight())
            properties['aromaticity'].append(analysis.aromaticity())
            properties['instability_index'].append(analysis.instability_index())
            properties['isoelectric_point'].append(analysis.isoelectric_point())
            properties['gravy'].append(analysis.gravy())
        except Exception as e:
            logger.warning(f"Error analyzing sequence: {e}")
    
    for prop, values in properties.items():
        if values:
            results[f'{prop}_mean'] = np.mean(values)
            results[f'{prop}_std'] = np.std(values)
    
    # 序列相似度计算（如果提供了原始序列）
    if original_sequences:
        similarities = []
        for gen_seq in sequences:
            gen_seq = gen_seq.rstrip('-')
            for orig_seq in original_sequences:
                if len(gen_seq) == 0 or len(orig_seq) == 0:
                    continue
                # 计算最长公共子序列相似度
                lcs_len = longest_common_subsequence(gen_seq, orig_seq)
                sim = lcs_len / max(len(gen_seq), len(orig_seq))
                similarities.append(sim)
        
        if similarities:
            results['similarity_mean'] = np.mean(similarities)
            results['similarity_std'] = np.std(similarities)
    
    return results

def longest_common_subsequence(s1, s2):
    """计算最长公共子序列长度"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def visualize_loss(loss_history, save_path):
    """可视化训练损失"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def visualize_aa_distribution(aa_freqs, save_path):
    """可视化氨基酸分布"""
    plt.figure(figsize=(12, 6))
    
    # 排序以便更好地可视化
    sorted_items = sorted(aa_freqs.items(), key=lambda x: x[1], reverse=True)
    aa_labels, freqs = zip(*sorted_items)
    
    sns.barplot(x=list(aa_labels), y=list(freqs))
    plt.title('Amino Acid Frequency Distribution')
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def custom_train_with_logging(model, diffusion, dataloader, optimizer, device, num_epochs=5, save_dir='models'):
    """带有详细日志记录的训练函数"""
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            # 移动数据到设备并转换为浮点类型
            batch = batch.float().to(device)
            
            # 采样时间步
            t = torch.randint(0, diffusion.num_timesteps, (batch.size(0),), device=device)
            
            # 采样噪声
            noise = torch.randn_like(batch)
            
            # 添加噪声到数据
            noisy_data, true_noise = diffusion.q_sample(batch, t, noise)
            
            # 预测噪声
            pred_noise = model(noisy_data, t)
            
            # 计算损失
            loss = F.mse_loss(pred_noise, true_noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_count += 1
            loss_history.append(batch_loss)
            
            pbar.set_postfix({'loss': batch_loss, 'avg_loss': epoch_loss / batch_count})
        
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss / batch_count,
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        logger.info(f'Epoch {epoch+1} average loss: {epoch_loss / batch_count}')
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss / batch_count,
    }, os.path.join(save_dir, 'final_model.pt'))
    
    return model, loss_history

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 读取测试数据
    with open(args.data_path, 'r') as f:
        test_sequences = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(test_sequences)} test sequences")
    
    if args.mode == 'train':
        # 创建数据集和数据加载器
        dataset = ProteinDataset(test_sequences, max_length=args.seq_length)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # 初始化模型和扩散过程
        model = ProteinDiffusionModel(
            seq_length=args.seq_length,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads
        ).to(args.device)
        
        diffusion = DiffusionProcess(num_timesteps=500, device=args.device)  # 减少时间步以加快调试，并传递设备参数
        
        # 初始化优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        # 训练模型
        model, loss_history = custom_train_with_logging(
            model,
            diffusion,
            dataloader,
            optimizer,
            args.device,
            num_epochs=args.num_epochs,
            save_dir=os.path.dirname(args.checkpoint_path)
        )
        
        # 可视化训练损失
        visualize_loss(loss_history, os.path.join('results', 'training_loss.png'))
        logger.info(f"Training completed. Model saved to {args.checkpoint_path}")
        
    elif args.mode == 'generate':
        # 加载模型
        model = load_model(
            args.checkpoint_path,
            args.seq_length,
            args.hidden_dim,
            args.num_layers,
            args.num_heads,
            args.device
        )
        
        diffusion = DiffusionProcess(num_timesteps=500, device=args.device)  # 减少时间步以加快调试，并传递设备参数
        
        # 生成序列
        generated_sequences = generate_sequences(
            model,
            diffusion,
            args.num_sequences,
            args.seq_length,
            args.device
        )
        
        # 保存生成的序列
        with open(args.output_path, 'w') as f:
            for i, seq in enumerate(generated_sequences):
                f.write(f'>Generated_sequence_{i+1}\n')
                f.write(f'{seq}\n')
        
        logger.info(f"Generated {len(generated_sequences)} sequences and saved to {args.output_path}")
        
        # 分析生成的序列
        analysis_results = analyze_sequences(generated_sequences, test_sequences)
        
        # 打印分析结果
        logger.info("Sequence Analysis Results:")
        for key, value in analysis_results.items():
            if key != 'aa_frequencies':
                logger.info(f"{key}: {value}")
        
        # 可视化氨基酸分布
        visualize_aa_distribution(
            analysis_results['aa_frequencies'], 
            os.path.join('../results', 'aa_distribution.png')
        )
        
    elif args.mode == 'analyze':
        # 读取生成的序列
        generated_sequences = []
        with open(args.output_path, 'r') as f:
            for line in f:
                if not line.startswith('>'):
                    generated_sequences.append(line.strip())
        
        # 分析生成的序列
        analysis_results = analyze_sequences(generated_sequences, test_sequences)
        
        # 打印分析结果
        logger.info("Sequence Analysis Results:")
        for key, value in analysis_results.items():
            if key != 'aa_frequencies':
                logger.info(f"{key}: {value}")
        
        # 可视化氨基酸分布
        visualize_aa_distribution(
            analysis_results['aa_frequencies'], 
            os.path.join('../results', 'aa_distribution.png')
        )

if __name__ == '__main__':
    main() 