import numpy as np
import pickle
import os


class LSTM:
    def __init__(self, vocab_size, hidden_size, seq_len):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        n = hidden_size + vocab_size
        self.Wf = np.random.randn(hidden_size, n) / np.sqrt(n)
        self.Wi = np.random.randn(hidden_size, n) / np.sqrt(n)
        self.Wc = np.random.randn(hidden_size, n) / np.sqrt(n)
        self.Wo = np.random.randn(hidden_size, n) / np.sqrt(n)
        self.Wv = np.random.randn(vocab_size, hidden_size) / np.sqrt(hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bv = np.zeros((vocab_size, 1))
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dWv = np.zeros_like(self.Wv)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)
        self.dbv = np.zeros_like(self.bv)
        self.m_Wf = np.zeros_like(self.Wf)
        self.v_Wf = np.zeros_like(self.Wf)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x.clip(-50, 50)))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward_step(self, x, h_prev, c_prev):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] != self.vocab_size:
            if x.shape[1] == self.vocab_size:
                x = x.T
            else:
                raise ValueError(f"输入x形状错误: {x.shape}")
        batch_size = x.shape[1]
        if h_prev.shape[1] != batch_size:
            h_prev = np.tile(h_prev[:, 0:1], (1, batch_size))
        if c_prev.shape[1] != batch_size:
            c_prev = np.tile(c_prev[:, 0:1], (1, batch_size))
        z = np.vstack((h_prev, x))
        a_f = self.Wf @ z + self.bf
        f = self.sigmoid(a_f)
        a_i = self.Wi @ z + self.bi
        i = self.sigmoid(a_i)
        a_c = self.Wc @ z + self.bc
        c_tilde = self.tanh(a_c)
        a_o = self.Wo @ z + self.bo
        o = self.sigmoid(a_o)
        c = f * c_prev + i * c_tilde
        h = o * self.tanh(c)
        v = self.Wv @ h + self.bv
        y = self.softmax(v)
        return y, h, c, (z, h_prev, c_prev, f, i, c_tilde, o, c, h, y)

    def sample(self, prime_text, length, temperature=1.0):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for char in prime_text:
            x = np.zeros((self.vocab_size, 1))
            x[char] = 1
            _, h, c, _ = self.forward_step(x, h, c)
        generated = list(prime_text)
        current_char = prime_text[-1] if prime_text else np.random.randint(0, self.vocab_size)
        for _ in range(length):
            x = np.zeros((self.vocab_size, 1))
            x[current_char] = 1
            y, h, c, _ = self.forward_step(x, h, c)
            epsilon = 1e-10
            min_prob = 1e-10
            y = np.maximum(y, min_prob)
            y = y / np.sum(y)
            y = np.log(y) / temperature
            y = np.exp(y - np.max(y))
            sum_y = np.sum(y)
            if sum_y < 1e-10:
                y = np.ones_like(y) / self.vocab_size
            else:
                y = y / sum_y
            current_char = np.random.choice(self.vocab_size, p=y.ravel())
            generated.append(current_char)
        return generated


def load_model(model_path):
    if not os.path.exists(model_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(current_dir, model_path.lstrip('/'))
        if os.path.exists(relative_path):
            model_path = relative_path
        else:
            possible_paths = [
                os.path.join(current_dir, 'outputs', os.path.basename(model_path)),
                os.path.join(current_dir, '..', 'outputs', os.path.basename(model_path)),
                os.path.join(current_dir, 'models', os.path.basename(model_path))
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"使用找到的模型路径: {model_path}")
                    break
    print(f"尝试加载模型文件: {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / 1024:.2f} KB")
    file_ext = os.path.splitext(model_path)[1].lower()
    try:
        if file_ext == '.npy':
            model_params = np.load(model_path, allow_pickle=True).item()
        else:
            with open(model_path, 'rb') as f:
                model_params = pickle.load(f)
        print("模型参数加载成功！")
        print(f"参数键: {list(model_params.keys())}")
        vocab_size = model_params['Wv'].shape[0]
        hidden_size = model_params['Wv'].shape[1]
        seq_len = 100
        model = LSTM(vocab_size, hidden_size, seq_len)
        model.Wf = model_params['Wf']
        model.Wi = model_params['Wi']
        model.Wc = model_params['Wc']
        model.Wo = model_params['Wo']
        model.Wv = model_params['Wv']
        model.bf = model_params['bf']
        model.bi = model_params['bi']
        model.bc = model_params['bc']
        model.bo = model_params['bo']
        model.bv = model_params['bv']
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("尝试使用备用方法加载...")
        try:
            data = np.load(model_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                if len(data) == 1 and isinstance(data[0], dict):
                    model_params = data[0]
                else:
                    model_params = {}
                    for i, key in enumerate(['Wf', 'Wi', 'Wc', 'Wo', 'Wv', 'bf', 'bi', 'bc', 'bo', 'bv']):
                        if i < len(data):
                            model_params[key] = data[i]
            else:
                model_params = {}
                for i, key in enumerate(['Wf', 'Wi', 'Wc', 'Wo', 'Wv', 'bf', 'bi', 'bc', 'bo', 'bv']):
                    if hasattr(data, key):
                        model_params[key] = getattr(data, key)
                    elif i < len(data):
                        model_params[key] = data[i]
            print("使用numpy加载成功！")
            print(f"参数键: {list(model_params.keys())}")
            vocab_size = model_params['Wv'].shape[0]
            hidden_size = model_params['Wv'].shape[1]
            seq_len = 100
            model = LSTM(vocab_size, hidden_size, seq_len)
            model.Wf = model_params['Wf']
            model.Wi = model_params['Wi']
            model.Wc = model_params['Wc']
            model.Wo = model_params['Wo']
            model.Wv = model_params['Wv']
            model.bf = model_params['bf']
            model.bi = model_params['bi']
            model.bc = model_params['bc']
            model.bo = model_params['bo']
            model.bv = model_params['bv']
            return model
        except Exception as e2:
            print(f"备用方法加载失败: {e2}")
            print("请检查模型文件是否损坏或格式是否正确。")
            raise


def load_char_mappings(data_path):
    if os.path.exists(data_path):
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            print(f"成功加载字符映射文件: {data_path}")
            print(f"数据类型: {type(data)}")
            if isinstance(data, dict):
                if 'char_to_idx' in data and 'idx_to_char' in data:
                    return data['char_to_idx'], data['idx_to_char']
                elif 'char2idx' in data and 'idx2char' in data:
                    return data['char2idx'], data['idx2char']
                elif 'chars' in data:
                    chars = data['chars']
                    char_to_idx = {ch: i for i, ch in enumerate(chars)}
                    idx_to_char = {i: ch for i, ch in enumerate(chars)}
                    return char_to_idx, idx_to_char
                else:
                    print(f"警告: 字典缺少必要键，可用键: {list(data.keys())}")
            elif isinstance(data, (tuple, list)):
                if len(data) >= 2:
                    return data[0], data[1]
                else:
                    print(f"警告: 元组/列表长度不足，需至少2个元素，实际{len(data)}个")
            elif isinstance(data, np.ndarray):
                if data.dtype == object:
                    chars = list(data)
                    char_to_idx = {ch: i for i, ch in enumerate(chars)}
                    idx_to_char = {i: ch for i, ch in enumerate(chars)}
                    return char_to_idx, idx_to_char
            print(f"错误: 无法解析字符映射格式，数据示例: {str(data)[:100]}...")
            print("使用默认ASCII字符映射")
        except Exception as e:
            print(f"加载映射文件出错: {e}")
            print("使用默认ASCII字符映射")
    print("使用默认ASCII字符映射")
    chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:\'"!?()[]{}<>+-*/=~_|')
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def generate_text(model, char_to_idx, idx_to_char, prime_text, length=200, temperature=1.0):
    prime_indices = [char_to_idx[char] for char in prime_text if char in char_to_idx]
    if not prime_indices:
        prime_indices = [np.random.randint(0, model.vocab_size)]
    generated_indices = model.sample(prime_indices, length, temperature)
    generated_text = ''.join([idx_to_char[idx] for idx in generated_indices])
    return generated_text


def main():
    model_path = 'outputs/model_epoch_10_batch_400.npy'
    data_path = 'outputs/char_mappings.pkl'
    print("加载模型...")
    try:
        model = load_model(model_path)
        print(f"模型加载成功! 词汇表大小: {model.vocab_size}, 隐藏层大小: {model.hidden_size}")
    except Exception as e:
        print(f"无法加载模型: {e}")
        return
    if os.path.exists(data_path):
        char_to_idx, idx_to_char = load_char_mappings(data_path)
    else:
        print(f"警告: 找不到字符映射文件 {data_path}")
        print("使用默认的ASCII字符映射")
        chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:\'"!?()[]{}<>+-*/=~_|')
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
    while True:
        try:
            prime_text = input("\n请输入提示文本 (输入'quit'退出): ")
            if prime_text.lower() == 'quit':
                break
            temperature = float(input("请输入温度参数 (默认1.0): ") or 1.0)
            length = int(input("请输入生成文本长度 (默认200): ") or 200)
            print("\n生成文本中...")
            generated_text = generate_text(model, char_to_idx, idx_to_char, prime_text, length, temperature)
            print("\n" + "-" * 50)
            print(generated_text)
            print("-" * 50)
        except Exception as e:
            print(f"生成文本时出错: {e}")
            print("请重试。")


if __name__ == "__main__":
    main()