import numpy as np


class LSTM:
    def __init__(self, vocab_size, hidden_size, seq_len):
        """初始化LSTM模型参数"""
        self.vocab_size = vocab_size  # 词汇表大小（字符数）
        self.hidden_size = hidden_size  # 隐藏层维度
        self.seq_len = seq_len  # 序列长度

        # 权重初始化（Xavier初始化）
        n = hidden_size + vocab_size
        self.Wf = np.random.randn(hidden_size, n) / np.sqrt(n)  # 遗忘门权重
        self.Wi = np.random.randn(hidden_size, n) / np.sqrt(n)  # 输入门权重
        self.Wc = np.random.randn(hidden_size, n) / np.sqrt(n)  # 细胞门权重
        self.Wo = np.random.randn(hidden_size, n) / np.sqrt(n)  # 输出门权重
        self.Wv = np.random.randn(vocab_size, hidden_size) / np.sqrt(hidden_size)  # 输出层权重

        # 偏置初始化
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bv = np.zeros((vocab_size, 1))

        # 梯度存储
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

        # Adam优化器参数
        self.m_Wf = np.zeros_like(self.Wf)
        self.v_Wf = np.zeros_like(self.Wf)
        # 其他参数的Adam状态...

    def sigmoid(self, x):
        """sigmoid激活函数，带数值稳定性处理"""
        return 1 / (1 + np.exp(-x.clip(-50, 50)))

    def tanh(self, x):
        """tanh激活函数"""
        return np.tanh(x)

    def softmax(self, x):
        """softmax激活函数，减去最大值保证数值稳定"""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward_step(self, x, h_prev, c_prev):
        """单时间步前向传播"""
        # 确保输入x是二维的 [vocab_size, batch_size]
        if x.ndim == 1:
            x = x.reshape(-1, 1)  # 转换为[vocab_size, 1]的形状

        # 检查维度并确保x的形状正确
        if x.shape[0] != self.vocab_size:
            # 如果第一个维度不是vocab_size，可能是转置问题
            if x.shape[1] == self.vocab_size:
                x = x.T  # 转置以匹配预期形状
            else:
                raise ValueError(f"输入x的形状不正确: {x.shape}, 预期第一个维度为{self.vocab_size}")

        batch_size = x.shape[1]  # 获取批次大小

        # 确保h_prev和c_prev的形状正确
        if h_prev.shape[1] != batch_size:
            # 如果h_prev的批次大小与输入不同，取第一列并扩展
            h_prev = np.tile(h_prev[:, 0:1], (1, batch_size))
        if c_prev.shape[1] != batch_size:
            c_prev = np.tile(c_prev[:, 0:1], (1, batch_size))

        # 拼接输入：[隐藏状态, 输入字符]
        z = np.vstack((h_prev, x))  # z shape: [hidden_size+vocab_size, batch_size]

        # 计算各门将
        a_f = self.Wf @ z + self.bf
        f = self.sigmoid(a_f)
        a_i = self.Wi @ z + self.bi
        i = self.sigmoid(a_i)
        a_c = self.Wc @ z + self.bc
        c_tilde = self.tanh(a_c)
        a_o = self.Wo @ z + self.bo
        o = self.sigmoid(a_o)

        # 细胞状态和隐藏状态更新
        c = f * c_prev + i * c_tilde
        h = o * self.tanh(c)

        # 输出层
        v = self.Wv @ h + self.bv
        y = self.softmax(v)

        # 保存中间变量用于反向传播
        cache = (z, h_prev, c_prev, f, i, c_tilde, o, c, h, y)
        return y, h, c, cache

    def backward_step(self, dJ_dy, h, c, cache, h_next_grad, c_next_grad):
        """单时间步反向传播"""
        z, h_prev, c_prev, f, i, c_tilde, o, c_curr, h_curr, y = cache
        vocab_size, hidden_size = self.Wv.shape
        batch_size = dJ_dy.shape[1]  # 获取批次大小

        # 确保dJ_dy的形状正确
        if dJ_dy.shape[0] != vocab_size:
            if dJ_dy.shape[1] == vocab_size:
                dJ_dy = dJ_dy.T  # 转置以匹配预期形状
            else:
                raise ValueError(f"dJ_dy的形状不正确: {dJ_dy.shape}, 预期第一个维度为{vocab_size}")

        # 输出层梯度
        dJ_dv = dJ_dy  # dJ/dv = dJ/dy (因v到y是softmax，导数为y - target)
        dJ_dWv = dJ_dv @ h.T
        dJ_dbv = np.sum(dJ_dv, axis=1, keepdims=True)

        # 隐藏状态梯度
        dJ_dh = self.Wv.T @ dJ_dv

        # 确保h_next_grad的形状正确
        if h_next_grad.shape[1] != batch_size:
            h_next_grad = np.tile(h_next_grad[:, 0:1], (1, batch_size))

        # 合并当前层和下一时间步的梯度
        dJ_dh += h_next_grad

        # 确保o和dtanh的形状正确
        if o.shape[1] != batch_size:
            o = np.tile(o[:, 0:1], (1, batch_size))

        dtanh = 1 - self.tanh(c) ** 2
        if dtanh.shape[1] != batch_size:
            dtanh = np.tile(dtanh[:, 0:1], (1, batch_size))

        # 确保c_next_grad的形状正确
        if c_next_grad.shape[1] != batch_size:
            c_next_grad = np.tile(c_next_grad[:, 0:1], (1, batch_size))

        # 细胞状态梯度
        dJ_dc = dJ_dh * o * dtanh + c_next_grad

        # 确保i的形状正确
        if i.shape[1] != batch_size:
            i = np.tile(i[:, 0:1], (1, batch_size))

        dJ_dc_tilde = dJ_dc * i

        # 确保c_tilde的形状正确
        if c_tilde.shape[1] != batch_size:
            c_tilde = np.tile(c_tilde[:, 0:1], (1, batch_size))

        da_c = dJ_dc_tilde * (1 - c_tilde ** 2)  # tanh导数
        dJ_dWc = da_c @ z.T
        dJ_dbc = np.sum(da_c, axis=1, keepdims=True)

        # 输入门梯度
        dJ_di = dJ_dc * c_tilde

        # 确保i的形状正确
        if i.shape[1] != batch_size:
            i = np.tile(i[:, 0:1], (1, batch_size))

        da_i = dJ_di * i * (1 - i)
        dJ_dWi = da_i @ z.T
        dJ_dbi = np.sum(da_i, axis=1, keepdims=True)

        # 遗忘门梯度
        dJ_df = dJ_dc * c_prev

        # 确保f的形状正确
        if f.shape[1] != batch_size:
            f = np.tile(f[:, 0:1], (1, batch_size))

        da_f = dJ_df * f * (1 - f)
        dJ_dWf = da_f @ z.T
        dJ_dbf = np.sum(da_f, axis=1, keepdims=True)

        # 输出门梯度
        dJ_do = dJ_dh * self.tanh(c)

        # 确保o的形状正确
        if o.shape[1] != batch_size:
            o = np.tile(o[:, 0:1], (1, batch_size))

        da_o = dJ_do * o * (1 - o)  # sigmoid导数
        dJ_dWo = da_o @ z.T
        dJ_dbo = np.sum(da_o, axis=1, keepdims=True)

        # 输入与历史状态梯度
        dJ_dz = (self.Wf.T @ da_f + self.Wi.T @ da_i +
                 self.Wo.T @ da_o + self.Wc.T @ da_c)
        dJ_dh_prev = dJ_dz[:hidden_size, :]
        dJ_dc_prev = dJ_dc * f

        # 累积梯度
        self.dWf += dJ_dWf
        self.dWi += dJ_dWi
        self.dWc += dJ_dWc
        self.dWo += dJ_dWo
        self.dWv += dJ_dWv
        self.dbf += dJ_dbf
        self.dbi += dJ_dbi
        self.dbc += dJ_dbc
        self.dbo += dJ_dbo
        self.dbv += dJ_dbv

        return dJ_dh_prev, dJ_dc_prev

    def reset_gradients(self):

        self.dWf.fill(0)
        self.dWi.fill(0)
        self.dWc.fill(0)
        self.dWo.fill(0)
        self.dWv.fill(0)
        self.dbf.fill(0)
        self.dbi.fill(0)
        self.dbc.fill(0)
        self.dbo.fill(0)
        self.dbv.fill(0)

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

            # 温度采样：temperature越低，分布越尖锐
            y = np.log(y) / temperature
            y = np.exp(y) / np.sum(np.exp(y))

            # 按概率选择下一个字符
            current_char = np.random.choice(self.vocab_size, p=y.ravel())
            generated.append(current_char)

        return generated
