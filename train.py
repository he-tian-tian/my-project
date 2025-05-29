import numpy as np
from lstm import LSTM
from trainer import LSTMTrainer
import pickle
import os
import time

# 配置参数
config = {
    'vocab_size': 0,  # 将在加载数据后设置
    'hidden_size': 128,  # 隐藏层维度
    'seq_len': 30,  # 序列长度
    'batch_size': 32,  # 批次大小
    'learning_rate': 0.01,  # 学习率
    'num_epochs': 10,  # 训练轮数
    'temperature': 0.8,  # 采样温度
    'checkpoint_every': 100,  # 保存检查点间隔
    'sample_every': 500  # 生成样本间隔
}


def load_data(file_path):
    """加载并预处理文本数据"""
    # 尝试不同的编码格式
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            print(f"无法使用 {encoding} 编码读取文件")
            if encoding == encodings[-1]:
                raise

    # 去重并排序字符
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # 构建映射
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # 转换为索引序列
    idx_sequence = np.array([char_to_idx[ch] for ch in text])

    return idx_sequence, char_to_idx, idx_to_char, vocab_size


def create_batches(sequence, seq_len, batch_size, vocab_size):
    """将序列分割为批次"""
    n_batches = len(sequence) // (seq_len * batch_size)
    sequence = sequence[:n_batches * seq_len * batch_size]

    # 重塑为[seq_len, batch_size]
    X = sequence.reshape(batch_size, -1).T.reshape(-1, seq_len, batch_size)
    y = np.roll(X, -1, axis=1)  # y是X的下一个字符

    # 转换为one-hot编码
    X_onehot = np.zeros((X.shape[0], X.shape[1], vocab_size))
    y_onehot = np.zeros((y.shape[0], y.shape[1], vocab_size))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_onehot[i, j, X[i, j]] = 1
            y_onehot[i, j, y[i, j]] = 1

    return X_onehot, y_onehot


def train():
    """训练LSTM模型"""
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)

    # 加载数据
    print("加载数据...")
    # 修改文件路径为正确的路径
    idx_sequence, char_to_idx, idx_to_char, vocab_size = load_data('HarryPotter_Stone.txt')
    config['vocab_size'] = vocab_size

    # 保存字符映射
    with open('outputs/char_mappings.pkl', 'wb') as f:
        pickle.dump((char_to_idx, idx_to_char), f)

    # 创建批次
    print("创建训练批次...")
    X, y = create_batches(idx_sequence, config['seq_len'], config['batch_size'], vocab_size)
    n_batches = X.shape[0]

    # 初始化模型和训练器
    print(f"初始化模型 (vocab_size={vocab_size}, hidden_size={config['hidden_size']})...")
    model = LSTM(vocab_size, config['hidden_size'], config['seq_len'])
    trainer = LSTMTrainer(model, learning_rate=config['learning_rate'])

    # 训练循环
    print(f"开始训练，共 {config['num_epochs']} 轮，每轮 {n_batches} 批次...")
    start_time = time.time()

    for epoch in range(config['num_epochs']):
        # 初始化状态
        h_prev = np.zeros((config['hidden_size'], config['batch_size']))
        c_prev = np.zeros((config['hidden_size'], config['batch_size']))

        epoch_loss = 0

        for batch in range(n_batches):
            # 获取当前批次
            X_batch = X[batch]  # [seq_len, batch_size, vocab_size]
            y_batch = y[batch]  # [seq_len, batch_size, vocab_size]

            # 训练一步
            loss, h_prev, c_prev = trainer.train_step(X_batch, y_batch, h_prev, c_prev)
            epoch_loss += loss

            # 打印进度
            if (batch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch + 1}/{config['num_epochs']}, Batch {batch + 1}/{n_batches}, "
                      f"Loss: {loss:.4f}, Time: {elapsed:.1f}s")

            # 保存检查点
            if (batch + 1) % config['checkpoint_every'] == 0:
                checkpoint_path = f"outputs/model_epoch_{epoch + 1}_batch_{batch + 1}.npy"
                save_model(model, checkpoint_path)
                print(f"保存检查点到: {checkpoint_path}")

            # 生成样本
            if (batch + 1) % config['sample_every'] == 0:
                sample_text(model, char_to_idx, idx_to_char, config['temperature'])

        # 每个epoch结束后
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch + 1}/{config['num_epochs']} 完成, 平均损失: {avg_loss:.4f}")

        # 每个epoch保存模型
        checkpoint_path = f"outputs/model_epoch_{epoch + 1}.npy"
        save_model(model, checkpoint_path)
        print(f"保存检查点到: {checkpoint_path}")

    # 训练结束后保存最终模型
    final_path = "outputs/model_params.npy"
    save_model(model, final_path)
    print(f"训练完成! 最终模型保存到: {final_path}")
    print(f"总训练时间: {time.time() - start_time:.1f}s")


def save_model(model, path):
    """保存模型参数"""
    params = {}
    for param_name in ['Wf', 'Wi', 'Wc', 'Wo', 'Wv', 'bf', 'bi', 'bc', 'bo', 'bv']:
        params[param_name] = getattr(model, param_name)
    np.save(path, params)


def sample_text(model, char_to_idx, idx_to_char, temperature):
    """生成样本文本"""
    # 选择一些起始文本
    start_texts = [
        "Harry Potter",
        "The Hogwarts Express",
        "Professor Dumbledore",
        "In the Chamber of Secrets",
        "Voldemort's wand"
    ]

    print("\n生成样本文本:")
    for start_text in start_texts:
        # 将文本转换为索引
        prime_indices = [char_to_idx.get(c, 0) for c in start_text]

        # 生成文本
        generated_indices = model.sample(prime_indices, 100, temperature)

        # 转换回字符
        generated_text = ''.join([idx_to_char[idx] for idx in generated_indices])

        print(f"起始: '{start_text}'")
        print(f"生成: {generated_text}\n")


if __name__ == "__main__":
    train()
