#!/bin/bash

# 创建必要的目录
mkdir -p models
mkdir -p results

# 训练模式 - 使用小数据集和较少的epoch进行快速调试
echo "=============== 运行训练模式 ==============="
python src/debug_model.py \
    --mode train \
    --data_path data/test_sequences.txt \
    --seq_length 500 \
    --hidden_dim 128 \
    --num_layers 4 \
    --num_heads 4 \
    --batch_size 2 \
    --num_epochs 5 \
    --checkpoint_path models/debug_checkpoint.pt

# 生成模式 - 生成新的序列并分析
echo "=============== 运行生成模式 ==============="
python src/debug_model.py \
    --mode generate \
    --checkpoint_path models/checkpoint_epoch_5.pt \
    --seq_length 500 \
    --hidden_dim 128 \
    --num_layers 4 \
    --num_heads 4 \
    --num_sequences 10 \
    --output_path results/generated_sequences.txt \
    --data_path data/test_sequences.txt

# 分析模式 - 对已生成的序列进行深入分析
echo "=============== 运行分析模式 ==============="
python src/debug_model.py \
    --mode analyze \
    --data_path data/test_sequences.txt \
    --output_path results/generated_sequences.txt

echo "运行完成！请查看 results 目录中的结果文件和图表。" 