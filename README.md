LSTM 文本生成模型
基于长短期记忆网络 (LSTM) 的文本生成项目，可根据给定文本前缀生成连续自然语言文本，适用于文本创作、语言模型学习等场景。
项目特点
完整 LSTM 实现：包含前向传播、反向传播及参数更新
文本生成功能：支持温度采样策略，控制生成文本的随机性
Adam 优化器：实现高效的参数更新策略
梯度裁剪：防止训练过程中梯度爆炸
检查点机制：训练过程中定期保存模型状态
项目结构
plaintext
lstm.py         # LSTM模型核心实现
trainer.py      # 训练器及优化器实现
train.py        # 训练脚本及数据处理
text.py         # 文本生成交互脚本
outputs/        # 模型参数及字符映射保存目录
环境要求
Python 3.6+
NumPy (主要依赖)
Pickle (用于保存模型和映射)
快速开始
安装依赖
bash
pip install numpy
准备训练数据
将训练文本保存为HarryPotter_Stone.txt
或修改train.py中load_data函数的文件路径
开始训练
bash
python train.py
文本生成
bash
python text.py
模型参数配置
在train.py中可配置以下参数：

python
运行
config = {
    'vocab_size': 0,       # 词汇表大小（自动根据数据设置）
    'hidden_size': 128,    # 隐藏层维度，影响模型表达能力
    'seq_len': 30,         # 输入序列长度，控制上下文窗口
    'batch_size': 32,      # 批次大小，影响训练稳定性
    'learning_rate': 0.01, # 学习率，控制参数更新步长
    'num_epochs': 10,      # 训练轮数
    'temperature': 0.8,    # 采样温度，值越小生成越确定
    'checkpoint_every': 100, # 检查点保存间隔
    'sample_every': 500    # 生成样本间隔
}
文本生成原理
生成过程通过以下步骤实现：

使用前缀文本初始化 LSTM 隐藏状态
对每个时间步：
将当前字符转换为 one-hot 编码
前向传播获取下一个字符的概率分布
应用温度采样策略从分布中选择字符
更新隐藏状态并重复

温度参数影响：

temperature < 1：分布更尖锐，倾向选择高概率字符
temperature = 1：标准分布
temperature > 1：分布更平滑，生成更随机
示例输出
plaintext
请输入提示文本 (输入'quit'退出): Harry Potter
请输入温度参数 (默认1.0): 0.8
请输入生成文本长度 (默认200): 150

生成文本中...
--------------------------------------------------
Harry Potter and the Sorcerer's Stone is a wonderful story about a young boy who discovers he is a wizard. He goes to Hogwarts School of Witchcraft and Wizardry, where he makes new friends and learns about magic. Along the way, he must face the evil Lord Voldemort, who killed his parents and wants to destroy him. With the help of his friends Ron and Hermione, Harry overcomes many challenges and learns the true meaning of friendship and bravery.
--------------------------------------------------
训练过程说明
数据预处理：将文本转换为字符索引序列，并分割为固定长度的批次
前向传播：按时间步计算各层输出并保存中间状态
反向传播：从最后一个时间步开始反向计算梯度
梯度裁剪：防止梯度爆炸，确保训练稳定
参数更新：使用 Adam 优化器更新模型参数
检查点保存：定期保存模型状态，支持断点续训
进阶使用
自定义训练数据
修改train.py中的load_data函数路径：

python
运行
idx_sequence, char_to_idx, idx_to_char, vocab_size = load_data('your_data.txt')
调整生成策略
在text.py中可修改温度参数以控制生成风格：

创意写作：temperature = 1.2
连贯续写：temperature = 0.6
加载预训练模型
修改text.py中的模型加载路径：

python
运行
model_path = 'outputs/model_epoch_10.npy'
