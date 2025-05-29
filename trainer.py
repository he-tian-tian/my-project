import numpy as np
from lstm import LSTM


class LSTMTrainer:
    def __init__(self, lstm_model, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """初始化训练器"""
        self.lstm = lstm_model
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adam优化器的动量和平方梯度
        self.m = {
            'Wf': np.zeros_like(lstm_model.Wf),
            'Wi': np.zeros_like(lstm_model.Wi),
            'Wc': np.zeros_like(lstm_model.Wc),
            'Wo': np.zeros_like(lstm_model.Wo),
            'Wv': np.zeros_like(lstm_model.Wv),
            'bf': np.zeros_like(lstm_model.bf),
            'bi': np.zeros_like(lstm_model.bi),
            'bc': np.zeros_like(lstm_model.bc),
            'bo': np.zeros_like(lstm_model.bo),
            'bv': np.zeros_like(lstm_model.bv)
        }
        self.v = {k: np.zeros_like(v) for k, v in self.m.items()}
        self.t = 0  # 时间步计数器

    def clip_gradients(self, max_norm=5.0):
        """梯度裁剪，防止梯度爆炸"""
        params = ['Wf', 'Wi', 'Wc', 'Wo', 'Wv', 'bf', 'bi', 'bc', 'bo', 'bv']
        grads = [getattr(self.lstm, f'd{param}') for param in params]

        # 计算全局梯度范数
        total_norm = 0
        for g in grads:
            param_norm = np.linalg.norm(g)
            total_norm += param_norm ** 2
        total_norm = np.sqrt(total_norm)

        # 裁剪梯度
        clip_coef = max_norm / (total_norm + self.epsilon)
        if clip_coef < 1:
            for param in params:
                grad = getattr(self.lstm, f'd{param}')
                setattr(self.lstm, f'd{param}', grad * clip_coef)

    def update_parameters(self):
        """使用Adam优化器更新参数"""
        self.t += 1
        params = ['Wf', 'Wi', 'Wc', 'Wo', 'Wv', 'bf', 'bi', 'bc', 'bo', 'bv']

        for param in params:
            grad = getattr(self.lstm, f'd{param}')
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

            # 偏差校正
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            # 参数更新
            setattr(self.lstm, param, getattr(self.lstm, param) -
                    self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon))

    def train_step(self, X, y, h_prev, c_prev):
        """执行一个批次的训练"""
        batch_size = X.shape[1]
        loss = 0
        h = h_prev
        c = c_prev
        h_grad = np.zeros_like(h_prev)
        c_grad = np.zeros_like(c_prev)

        # 重置梯度
        self.lstm.reset_gradients()

        # 前向传播与反向传播
        for t in range(X.shape[0]):
            x = X[t]  # 获取当前时间步的输入
            target = y[t]  # 获取当前时间步的目标

            # 确保输入和目标是二维的
            if x.ndim == 1:
                x = x.reshape(1, -1)  # 转换为[1, vocab_size]
            if target.ndim == 1:
                target = target.reshape(1, -1)  # 转换为[1, vocab_size]

            # 确保批次大小一致
            if x.shape[0] != batch_size:
                # 如果是单样本，扩展到批次大小
                if x.shape[0] == 1:
                    x = np.tile(x, (batch_size, 1))
                else:
                    # 否则，只取前batch_size个样本
                    x = x[:batch_size]

            if target.shape[0] != batch_size:
                if target.shape[0] == 1:
                    target = np.tile(target, (batch_size, 1))
                else:
                    target = target[:batch_size]

            # 前向传播
            y_pred, h, c, cache = self.lstm.forward_step(x.T, h, c)

            # 计算损失（交叉熵）
            loss += -np.sum(target * np.log(y_pred + 1e-10)) / batch_size

            # 反向传播
            dJ_dy = y_pred - target.T  # [vocab_size, batch_size]
            h_grad, c_grad = self.lstm.backward_step(
                dJ_dy, h, c, cache, h_grad, c_grad
            )

        # 梯度裁剪
        self.clip_gradients()

        # 更新参数
        self.update_parameters()

        return loss / X.shape[0], h, c
