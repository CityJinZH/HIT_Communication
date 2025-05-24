import numpy as np  # 导入NumPy库
from tqdm import tqdm  # 导入tqdm库
import config  # 导入配置文件
import os  # 导入os模块，用于文件操作
import copy  # 导入copy模块，用于深拷贝


class DNN_Numpy_Equalizer:  # 定义NumPy版DNN均衡器类
    def __init__(self, input_size, hidden_sizes, output_size=1,
                 dropout_rate=0.0, random_seed=None):  # 添加dropout_rate参数
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate  # 保存dropout率
        self.random_seed = random_seed
        self.parameters = {}
        self.num_layers = len(hidden_sizes) + 1  # 网络层数 (隐藏层+输出层)

        if self.random_seed is not None:  # 如果设置了随机种子
            np.random.seed(self.random_seed)  # 设置NumPy随机种子

        self._initialize_parameters()  # 初始化网络参数

    def _initialize_parameters(self):  # 初始化网络参数
        layer_dims = [self.input_size] + self.hidden_sizes + [self.output_size]  # 定义各层维度
        for l in range(1, len(layer_dims)):  # 遍历每一层
            if l == self.num_layers:  # 如果是输出层
                scale_factor = np.sqrt(1.0 / layer_dims[l - 1])  # Xavier for tanh
            else:  # 如果是隐藏层
                scale_factor = np.sqrt(2.0 / layer_dims[l - 1])  # He for ReLU

            self.parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * scale_factor  # 初始化权重
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))  # 初始化偏置
            self.parameters[f'W{l}'] = self.parameters[f'W{l}'].astype(np.float32)  # 类型转换
            self.parameters[f'b{l}'] = self.parameters[f'b{l}'].astype(np.float32)  # 类型转换

    def _relu(self, Z):  # ReLU激活函数
        return np.maximum(0, Z)

    def _relu_backward(self, dA, Z_cache):  # ReLU反向传播
        dZ = np.array(dA, copy=True)
        dZ[Z_cache <= 0] = 0
        return dZ

    def _tanh(self, Z):  # tanh激活函数
        return np.tanh(Z)

    def _tanh_backward(self, dA, Z_cache):  # tanh反向传播
        tanh_Z = np.tanh(Z_cache)
        dZ = dA * (1 - tanh_Z ** 2)
        return dZ

    def _forward_propagation(self, X_batch, training_mode=True):  # 前向传播，添加training_mode
        """
        training_mode (bool): True时启用Dropout，False时禁用
        """
        caches = []  # 初始化缓存
        dropout_masks = {}  # 初始化dropout掩码字典
        A = X_batch.T  # 转置输入 (features, samples)

        # 隐藏层
        for l in range(1, self.num_layers):  # 遍历隐藏层
            A_prev = A
            Wl = self.parameters[f'W{l}']
            bl = self.parameters[f'b{l}']

            Zl = np.dot(Wl, A_prev) + bl  # 线性计算
            Al = self._relu(Zl)  # ReLU激活

            if self.dropout_rate > 0 and training_mode:  # 如果启用dropout且在训练模式
                # 应用 inverted dropout
                Dl = np.random.rand(Al.shape[0], Al.shape[1]) < (1 - self.dropout_rate)  # 生成dropout掩码
                Al = Al * Dl / (1 - self.dropout_rate)  # 应用掩码并缩放 (inverted dropout)
                dropout_masks[f'D{l}'] = Dl  # 保存掩码

            cache = (A_prev, Wl, bl, Zl)  # 保存缓存
            caches.append(cache)
            A = Al  # 更新A

        # 输出层
        A_prev = A
        WL = self.parameters[f'W{self.num_layers}']
        bL = self.parameters[f'b{self.num_layers}']
        ZL = np.dot(WL, A_prev) + bL  # 线性计算
        AL = self._tanh(ZL)  # tanh激活
        cache = (A_prev, WL, bL, ZL)  # 保存缓存
        caches.append(cache)

        return AL.T, caches, dropout_masks  # 返回预测、缓存和dropout掩码

    def _compute_cost(self, AL, Y_batch):  # 计算损失
        m = Y_batch.shape[0]
        cost = (1 / m) * np.sum(np.square(AL - Y_batch))
        return cost

    def _backward_propagation(self, AL, Y_batch, caches, dropout_masks):  # 反向传播，添加dropout_masks
        grads = {}  # 初始化梯度
        m = AL.shape[0]
        AL_T = AL.T
        Y_batch_T = Y_batch.T

        # 输出层梯度
        current_cache = caches[self.num_layers - 1]
        A_prev_L, WL, bL, ZL_cache = current_cache
        dZL = (2 / m) * (AL_T - Y_batch_T) * (1 - AL_T ** 2)  # dL/dZL for MSE with tanh output
        grads[f'dW{self.num_layers}'] = np.dot(dZL, A_prev_L.T)  # dWL
        grads[f'db{self.num_layers}'] = np.sum(dZL, axis=1, keepdims=True)  # dbL
        dA_prev = np.dot(WL.T, dZL)  # dA for previous layer

        # 隐藏层梯度 (从后向前)
        for l in reversed(range(1, self.num_layers)):  # l from L-1 to 1
            current_cache = caches[l - 1]
            A_prev, Wl, bl, Zl_cache = current_cache

            # 应用dropout的反向传播 (如果该层有dropout)
            if self.dropout_rate > 0 and f'D{l}' in dropout_masks:  # 如果有dropout掩码
                Dl = dropout_masks[f'D{l}']  # 获取掩码
                dA_prev = dA_prev * Dl / (1 - self.dropout_rate)  # 应用相同的掩码和缩放

            dZl = self._relu_backward(dA_prev, Zl_cache)  # dZl for ReLU
            grads[f'dW{l}'] = np.dot(dZl, A_prev.T)  # dWl
            grads[f'db{l}'] = np.sum(dZl, axis=1, keepdims=True)  # dbl
            if l > 1:  # 如果不是第一个隐藏层
                dA_prev = np.dot(Wl.T, dZl)  # 计算传递给更前一层的dA

        return grads  # 返回梯度

    def _update_parameters(self, grads, learning_rate):  # 更新参数 (简单梯度下降)
        for l in range(1, self.num_layers + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, initial_learning_rate,
              lr_decay_epochs=None, lr_decay_factor=None,
              early_stopping_patience=None, min_delta=0.0):  # 添加早停和LR衰减参数
        """
        训练DNN模型 (NumPy手动实现)，加入早停和学习率衰减
        """
        num_samples = X_train.shape[0]  # 训练样本数
        num_batches = (num_samples + batch_size - 1) // batch_size  # 每轮批次数

        print(f"开始使用NumPy进行DNN训练，共 {epochs} 轮 (带早停和LR衰减)...")  # 打印训练信息

        best_val_loss = float('inf')  # 初始化最佳验证损失
        epochs_no_improve = 0  # 初始化未改善轮数
        current_lr = initial_learning_rate  # 初始化当前学习率
        best_parameters = None  # 用于保存最佳参数

        for epoch in range(epochs):  # 遍历每一轮
            epoch_loss = 0.0  # 初始化轮损失
            permutation = np.random.permutation(num_samples)  # 打乱数据
            shuffled_X = X_train[permutation]
            shuffled_y = y_train[permutation]

            # 学习率衰减
            if lr_decay_epochs and lr_decay_factor and (epoch + 1) % lr_decay_epochs == 0:  # 如果达到衰减条件
                current_lr *= lr_decay_factor  # 衰减学习率
                print(f"Epoch {epoch + 1}: Learning rate decayed to {current_lr:.6f}")  # 打印衰减信息

            batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs} (LR:{current_lr:.1e})",
                              unit="batch", leave=False)  # 创建批次进度条
            for i in batch_pbar:  # 遍历每一批次
                start_idx = i * batch_size  # 起始索引
                end_idx = min((i + 1) * batch_size, num_samples)  # 结束索引
                X_batch = shuffled_X[start_idx:end_idx]  # 当前批次输入
                y_batch = shuffled_y[start_idx:end_idx]  # 当前批次目标

                AL, caches, dropout_masks = self._forward_propagation(X_batch, training_mode=True)  # 前向传播
                batch_loss = self._compute_cost(AL, y_batch)  # 计算损失
                epoch_loss += batch_loss * X_batch.shape[0]  # 累加损失
                grads = self._backward_propagation(AL, y_batch, caches, dropout_masks)  # 反向传播
                self._update_parameters(grads, current_lr)  # 更新参数 (或 _update_parameters_adam)

                batch_pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})  # 更新进度条

            avg_epoch_loss = epoch_loss / num_samples  # 计算平均轮损失

            # 在验证集上评估
            AL_val, _, _ = self._forward_propagation(X_val, training_mode=False)  # 验证模式，不使用dropout
            val_loss = self._compute_cost(AL_val, y_val)  # 计算验证损失
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {val_loss:.4f}")  # 打印损失信息

            # 早停逻辑
            if early_stopping_patience:  # 如果启用早停
                if val_loss < best_val_loss - min_delta:  # 如果验证损失改善
                    best_val_loss = val_loss  # 更新最佳验证损失
                    epochs_no_improve = 0  # 重置未改善轮数
                    best_parameters = copy.deepcopy(self.parameters)  # 深拷贝当前最佳参数
                    # print(f"  Validation loss improved. Saved best parameters.") # 打印信息
                else:  # 如果验证损失未改善
                    epochs_no_improve += 1
                    # print(f"  Validation loss did not improve for {epochs_no_improve} epochs.")

                if epochs_no_improve >= early_stopping_patience:  # 如果达到早停条件
                    print(
                        f"Early stopping triggered after {epoch + 1} epochs. Best validation loss: {best_val_loss:.4f}")  # 打印早停信息
                    if best_parameters:  # 如果有保存的最佳参数
                        self.parameters = best_parameters  # 恢复最佳参数
                        print("Restored best parameters.")  # 打印恢复信息
                    break  # 跳出循环

        if not early_stopping_patience or epochs_no_improve < early_stopping_patience:  # 如果未触发早停
            print("Training finished (or reached max epochs).")  # 打印完成信息
            if best_parameters and val_loss > best_val_loss:  # 如果最后模型不是最好的
                self.parameters = best_parameters  # 确保使用最佳参数
                print("Final model parameters set to best validation performance parameters.")

    def predict(self, X_test):  # 预测函数
        AL, _, _ = self._forward_propagation(X_test, training_mode=False)  # 预测模式，不使用dropout
        return AL

    def predict_symbols(self, X_test):  # 预测符号函数
        raw_predictions = self.predict(X_test)
        symbols = np.sign(raw_predictions)
        symbols[symbols == 0] = 1
        return symbols.reshape(-1, 1)

