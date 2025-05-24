import numpy as np  # 导入NumPy库，用于数值计算
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import config  # 导入配置文件


class LMS_Equalizer:  # 定义LMS均衡器类
    def __init__(self, num_taps, mu, random_seed=None):
        """
        初始化LMS均衡器
        参数:
            num_taps (int): 均衡器抽头数量 (滤波器长度，通常等于窗口大小)
            mu (float): LMS算法的步长因子 (学习率)
            random_seed (int, optional): 随机种子 (LMS通常是确定性的，但若有随机初始化则有用)
        """
        self.num_taps = num_taps  # 保存抽头数量
        self.mu = mu  # 保存步长
        self.weights = np.zeros(num_taps, dtype=np.float32)  # 初始化权重为全零

        if random_seed is not None:
            np.random.seed(random_seed)  # 设置NumPy随机种子

    def train(self, X_train_windowed, y_train_desired, epochs=1):
        """
        训练LMS均衡器 (自适应过程)
        参数:
            X_train_windowed (numpy.ndarray): 训练输入数据，每行是一个接收信号窗口
                                              shape: (num_samples, num_taps)
            y_train_desired (numpy.ndarray): 训练期间的期望输出 (原始发送符号)
                                             shape: (num_samples,) or (num_samples, 1)
            epochs (int): 在整个训练数据集上迭代的次数
        """
        y_train_flat = y_train_desired.flatten()  # 确保y_train_desired是一维的
        num_samples = X_train_windowed.shape[0]  # 获取样本数量

        print(f"开始LMS均衡器训练/自适应过程，共 {epochs} 轮...")  # 打印训练开始信息

        for epoch in range(epochs):  # 遍历每一轮
            epoch_mse = 0.0  # 初始化当前轮的均方误差
            # 使用tqdm显示样本处理进度
            sample_pbar = tqdm(range(num_samples), desc=f"LMS Epoch {epoch + 1}/{epochs}", unit="sample",
                               leave=False)  # 创建样本进度条

            for i in sample_pbar:  # 遍历每个训练样本
                x_n = X_train_windowed[i, :]  # 当前输入窗口 x(n)
                d_n = y_train_flat[i]  # 当前期望输出 d(n)

                # 1. 计算均衡器输出 y_hat(n) = w^T(n) * x(n)
                y_hat_n = np.dot(self.weights, x_n)  # 计算均衡器输出

                # 2. 计算误差 e(n) = d(n) - y_hat(n)
                e_n = d_n - y_hat_n  # 计算误差
                epoch_mse += e_n ** 2  # 累加平方误差

                # 3. 更新权重 w(n+1) = w(n) + mu * e(n) * x(n)
                self.weights += self.mu * e_n * x_n  # 更新权重

                if i % (num_samples // 100 + 1) == 0:  # 每处理一定数量样本后更新进度条描述
                    sample_pbar.set_postfix({"current_error": f"{e_n:.4f}"})  # 更新进度条显示当前误差

            avg_epoch_mse = epoch_mse / num_samples  # 计算当前轮的平均均方误差
            print(f"LMS Epoch {epoch + 1}/{epochs} - Average Training MSE: {avg_epoch_mse:.4f}")  # 打印轮次信息和MSE
            print(f"LMS最终权重: {self.weights}")  # 打印最终权重

    def predict(self, X_test_windowed):
        """
        使用训练好的LMS均衡器权重进行预测
        参数:
            X_test_windowed (numpy.ndarray): 测试输入数据，每行是一个接收信号窗口
                                             shape: (num_samples, num_taps)
        返回:
            numpy.ndarray: 均衡器的输出 (估计的符号值)
        """
        num_samples = X_test_windowed.shape[0]  # 获取测试样本数量
        y_predicted = np.zeros(num_samples, dtype=np.float32)  # 初始化预测输出数组

        # 使用tqdm显示预测进度
        test_pbar = tqdm(range(num_samples), desc="LMS Predicting", unit="sample", leave=False)  # 创建预测进度条
        for i in test_pbar:  # 遍历每个测试样本
            x_n = X_test_windowed[i, :]  # 当前输入窗口
            y_predicted[i] = np.dot(self.weights, x_n)  # 计算预测输出

        return y_predicted.reshape(-1, 1)  # 返回预测结果，调整为列向量

    def predict_symbols(self, X_test_windowed):
        """
        使用LMS均衡器进行预测，并直接输出判决后的符号 (-1或1)
        参数:
            X_test_windowed (numpy.ndarray): 测试输入数据
        返回:
            numpy.ndarray: 判决后的BPSK符号 (-1或1)
        """
        raw_predictions = self.predict(X_test_windowed)  # 获取原始预测值
        symbols = np.sign(raw_predictions)  # 使用sign函数进行判决
        symbols[symbols == 0] = 1  # 将可能的0值映射为1
        return symbols.reshape(-1, 1)  # 返回判决后的符号，确保是列向量

