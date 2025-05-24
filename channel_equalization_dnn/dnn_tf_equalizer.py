import tensorflow as tf  # 导入TensorFlow库
import numpy as np  # 导入NumPy库
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import config  # 导入配置文件

if not tf.__version__.startswith('1.'):  # 检查TensorFlow版本
    tf = tf.compat.v1
    tf.disable_eager_execution()  # 禁用Eager Execution


class DNN_TF_Equalizer:  # 定义基于TensorFlow的DNN均衡器类
    def __init__(self, input_size, hidden_sizes, output_size=1,
                 dropout_rate=0.0, random_seed=None):  # 添加dropout_rate参数
        """
        初始化DNN均衡器模型 (TensorFlow实现)
        参数:
            input_size (int): 输入层神经元数量
            hidden_sizes (list of int): 每个隐藏层的神经元数量列表
            output_size (int): 输出层神经元数量
            dropout_rate (float): Dropout比率 (0到1之间)
            random_seed (int, optional): 随机种子
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate  # 保存dropout率
        self.random_seed = random_seed

        tf.reset_default_graph()  # 重置默认图
        if self.random_seed is not None:  # 如果设置了随机种子
            tf.set_random_seed(self.random_seed)  # 设置TF随机种子
            np.random.seed(self.random_seed)  # 设置NumPy随机种子

        self._build_model()  # 构建模型
        self.sess = tf.Session()  # 创建TF会话
        self.sess.run(tf.global_variables_initializer())  # 初始化变量
        self.saver = tf.train.Saver()  # 初始化Saver用于保存最佳模型 (早停用)

    def _build_model(self):
        """
        构建DNN的计算图
        """
        self.X_placeholder = tf.placeholder(tf.float32, [None, self.input_size], name="X_input")  # 输入占位符
        self.y_placeholder = tf.placeholder(tf.float32, [None, self.output_size], name="y_target")  # 目标占位符
        self.is_training_placeholder = tf.placeholder(tf.bool, name="is_training")  # 训练标志占位符 (用于Dropout)
        self.current_learning_rate_placeholder = tf.placeholder(tf.float32, shape=[],
                                                                name="current_learning_rate")  # 学习率占位符

        layer = self.X_placeholder  # 初始化当前层为输入

        for i, num_neurons in enumerate(self.hidden_sizes):  # 遍历隐藏层
            layer = tf.layers.dense(inputs=layer,
                                    units=num_neurons,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.glorot_uniform_initializer(
                                        seed=self.random_seed + i if self.random_seed else None),  # Xavier 初始化
                                    name=f"hidden_layer_{i + 1}")  # 定义全连接层
            if self.dropout_rate > 0:  # 如果dropout率大于0
                layer = tf.layers.dropout(inputs=layer,
                                          rate=self.dropout_rate,
                                          training=self.is_training_placeholder,  # 只有在训练时才使用dropout
                                          seed=self.random_seed + 100 + i if self.random_seed else None,  # dropout种子
                                          name=f"dropout_{i + 1}")  # 定义dropout层

        self.output_logits = tf.layers.dense(inputs=layer,
                                             units=self.output_size,
                                             activation=None,
                                             kernel_initializer=tf.glorot_uniform_initializer(
                                                 seed=self.random_seed + len(
                                                     self.hidden_sizes) if self.random_seed else None),  # Xavier 初始化
                                             name="output_logits")  # 定义输出层
        self.predictions = tf.tanh(self.output_logits, name="predictions")  # tanh激活

        self.loss = tf.reduce_mean(tf.square(self.predictions - self.y_placeholder), name="loss_mse")  # MSE损失

        optimizer = tf.train.AdamOptimizer(learning_rate=self.current_learning_rate_placeholder)  # Adam优化器
        self.train_op = optimizer.minimize(self.loss)  # 训练操作

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, initial_learning_rate,
              lr_decay_epochs=None, lr_decay_factor=None,  # 学习率衰减参数
              early_stopping_patience=None, min_delta=0.0):  # 早停参数
        """
        训练DNN模型，加入早停和学习率衰减
        """
        num_samples = X_train.shape[0]  # 训练样本数
        num_batches = (num_samples + batch_size - 1) // batch_size  # 每轮批次数

        print(f"开始使用TensorFlow进行DNN训练，共 {epochs} 轮 (带早停和LR衰减)...")  # 打印训练信息

        best_val_loss = float('inf')  # 初始化最佳验证损失
        epochs_no_improve = 0  # 初始化验证损失未改善的轮数
        current_lr = initial_learning_rate  # 初始化当前学习率

        best_model_path = "./tf_best_model.ckpt"  # 定义最佳模型保存路径

        for epoch in range(epochs):  # 遍历每一轮
            epoch_loss = 0.0  # 初始化轮损失
            permutation = np.random.permutation(num_samples)  # 打乱数据
            shuffled_X = X_train[permutation]
            shuffled_y = y_train[permutation]

            # 学习率衰减
            if lr_decay_epochs and lr_decay_factor and (epoch + 1) % lr_decay_epochs == 0:  # 如果达到衰减条件
                current_lr *= lr_decay_factor  # 衰减学习率
                print(f"Epoch {epoch + 1}: Learning rate decayed to {current_lr:.6f}")  # 打印学习率衰减信息

            batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs} (LR:{current_lr:.1e})",
                              unit="batch", leave=False)  # 创建批次进度条
            for i in batch_pbar:  # 遍历每一批次
                start_idx = i * batch_size  # 起始索引
                end_idx = min((i + 1) * batch_size, num_samples)  # 结束索引
                X_batch = shuffled_X[start_idx:end_idx]  # 当前批次输入
                y_batch = shuffled_y[start_idx:end_idx]  # 当前批次目标

                feed_dict = {  # 构建feed_dict
                    self.X_placeholder: X_batch,
                    self.y_placeholder: y_batch,
                    self.is_training_placeholder: True,  # 训练模式
                    self.current_learning_rate_placeholder: current_lr
                }
                _, batch_loss_val = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)  # 运行训练和损失计算
                epoch_loss += batch_loss_val * X_batch.shape[0]  # 累加批次损失
                batch_pbar.set_postfix({"batch_loss": f"{batch_loss_val:.5f}"})  # 更新进度条

            avg_epoch_loss = epoch_loss / num_samples  # 计算平均轮损失

            # 在验证集上评估
            val_feed_dict = {  # 构建验证集feed_dict
                self.X_placeholder: X_val,
                self.y_placeholder: y_val,
                self.is_training_placeholder: False  # 非训练模式 (Dropout不生效)
            }
            val_loss = self.sess.run(self.loss, feed_dict=val_feed_dict)  # 计算验证损失
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {val_loss:.5f}")  # 打印损失信息

            # 早停逻辑
            if early_stopping_patience:  # 如果启用了早停
                if val_loss < best_val_loss - min_delta:  # 如果验证损失改善
                    best_val_loss = val_loss  # 更新最佳验证损失
                    epochs_no_improve = 0  # 重置未改善轮数
                    self.saver.save(self.sess, best_model_path)  # 保存当前最佳模型
                    # print(f"  Validation loss improved. Saved model to {best_model_path}") # 打印模型保存信息
                else:  # 如果验证损失未改善
                    epochs_no_improve += 1  # 增加未改善轮数
                    # print(f"  Validation loss did not improve for {epochs_no_improve} epochs.") # 打印未改善信息

                if epochs_no_improve >= early_stopping_patience:  # 如果达到早停条件
                    print(
                        f"Early stopping triggered after {epoch + 1} epochs. Best validation loss: {best_val_loss:.4f}")  # 打印早停信息
                    self.saver.restore(self.sess, best_model_path)  # 恢复最佳模型权重
                    print(f"Restored best model from {best_model_path}")  # 打印模型恢复信息
                    break  # 跳出训练循环

        if not early_stopping_patience or epochs_no_improve < early_stopping_patience:  # 如果没有早停或未触发早停
            print("Training finished (or reached max epochs). Using last model state.")  # 打印训练完成信息
            # 保存最终模型
            # self.saver.save(self.sess, "./tf_final_model.ckpt")

    def predict(self, X_test):
        """
        使用训练好的模型进行预测
        """
        feed_dict = {self.X_placeholder: X_test, self.is_training_placeholder: False}  # 构建预测feed_dict (非训练模式)
        predictions = self.sess.run(self.predictions, feed_dict=feed_dict)  # 进行预测
        return predictions  # 返回预测结果

    def predict_symbols(self, X_test):
        """
        预测并输出判决符号
        """
        raw_predictions = self.predict(X_test)  # 获取原始预测
        symbols = np.sign(raw_predictions)  # 判决
        symbols[symbols == 0] = 1  # 处理0值
        return symbols  # 返回判决符号

    def close_session(self):
        """
        关闭TensorFlow会话
        """
        if self.sess:  # 如果会话存在
            self.sess.close()  # 关闭会话
            # print("TensorFlow session closed.") # 打印关闭信息 (可选)


