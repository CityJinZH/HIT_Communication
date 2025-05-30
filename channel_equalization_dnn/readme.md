**项目说明：**

1. **数据生成 (`data_generator.py`)**:
   - 生成随机的二进制比特序列。
   - 进行BPSK调制（0 -> -1, 1 -> +1）。
   - 模拟一个多径信道，引入符号间干扰 (ISI)。
   - 加入高斯白噪声 (AWGN)，通过SNR控制噪声强度。
   - **关键**：为均衡器准备输入。均衡器的输入通常是接收信号的一个滑动窗口 `[r(k-N), ..., r(k), ..., r(k+N)]`，其目标输出是原始发送符号 `s(k)`。
2. **DNN均衡器设计思想 (`dnn_tf_equalizer.py`, `dnn_numpy_equalizer.py`)**:
   - **输入层**: 神经元数量等于滑动窗口大小。输入是接收到的、经过信道和噪声影响的信号样本。
   - **隐藏层**:
     - 通常使用3到5个隐藏层。
     - 神经元数量可以逐渐减少，形成一种信息压缩和特征提取的结构（例如，`Input -> 64 -> 32 -> Output`）。
     - 激活函数：ReLU (`Rectified Linear Unit`) 是常用的选择，因为它计算简单且能缓解梯度消失问题。
   - **输出层**:
     - 1个神经元。
     - 激活函数：`tanh`。因为BPSK信号是-1或+1，`tanh`的输出范围是(-1, 1)，非常适合直接估计原始BPSK符号或其软信息。
   - **损失函数**: 均方误差 (MSE)。因为我们希望网络的输出尽可能接近原始的-1或+1。
3. **LMS均衡器 (`lms_equalizer.py`)**:
   - 经典的自适应滤波算法。
   - 输入也是接收信号的滑动窗口。
   - 通过最小化均方误差来迭代更新滤波器（均衡器）的权重。
4. **模块化子程序**:
   - 每个 `.py` 文件负责一个特定的功能。
   - `config.py` 集中管理超参数，方便调试和修改。
5. **程序会话选择与进度**:
   - `main.py` 提供运行时选择DNN实现方式。
   - 使用 `tqdm` 库显示训练和测试的进度条。
6. **性能比较**:
   - 在不同的信噪比 (SNR) 条件下测试各算法的误码率 (BER)。
   - 绘制BER vs SNR曲线图进行比较。

**注意事项**:

- **数据量**: 要达到高误码率性能，例如要达到10<sup>-4</sup>的BER，就意味着每10000个符号中最多只允许1个错误。测试时，需要远大于10000个符号才能得到可靠的BER统计。

- **信道模型**: 信道的恶劣程度会直接影响BER。一个非常简单的信道可能很容易均衡，而一个有深度衰落或长时延扩展的信道则非常困难。

- **超参数**: DNN的层数、神经元数、学习率、批大小、训练轮数，都需要仔细调整。

- **窗口大小**: 均衡器输入窗口的大小对性能至关重要。太小则包含的信息不足以消除ISI，太大则会引入更多噪声并增加计算复杂度。

channel_equalization/<br>
├── main.py                     # 主程序入口，负责流程控制、用户选择、结果绘图<br>
├── config.py                   # 配置文件，存放各种超参数和设置<br>
├── data_generator.py           # 数据生成模块 (BPSK调制, 信道模拟, 加噪声)<br>
├── dnn_tf_equalizer.py         # 基于TensorFlow的DNN均衡器实现<br>
├── dnn_numpy_equalizer.py      # 基于NumPy的DNN均衡器实现 (手动搭建)<br>
├── lms_equalizer.py            # LMS均衡算法实现<br>
├── metrics.py                  # 性能评估模块 (BER计算, 绘图)<br>
└── README.md                   # 程序说明 

  **如何运行：**

  1. **环境准备**:

     - 安装anaconda
       官网网址：https://www.anaconda.com/download/success
       镜像网址：https://mirror.tuna.tsinghua.edu.cn/help/anaconda/

     - 配置系统环境 path
     - 验证conda --version 

     - 安装 Python (推荐3.6-3.8  本项目使用3.7.16)。

     - 安装必要的库:

       ```
       pip install numpy==1.19.5 tqdm matplotlib
       pip install tensorflow-cpu==1.15.0 # 或者 tensorflow==1.15.0 
       ```
       
      具体可见: requirements.txt

       ```
       pip install -r requirements.txt
       若安装速度慢，可使用国内源
       -i https://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host
       -i https://mirrors.aliyun.com/pypi/simple  --trusted-host
       -i https://mirrors.cloud.tencent.com/pypi/simple  --trusted-host
       -i https://mirrors.huaweicloud.com/repository/pypi/simple  --trusted-host  
       ```
       
  2. **执行主程序:**

       ```
       bash 
       python main.py
       ```

  - 程序提示选择DNN的实现方式 (TensorFlow 或 NumPy)。

  **程序进度与时间统计：**

  - `tqdm` 库已经用于显示训练和测试的进度条。
  - `main.py` 的末尾会打印整个程序的总运行时间。


