import numpy as np  # 导入NumPy库，用于数值计算
import config  # 导入配置文件
from tqdm import tqdm

def generate_bpsk_signal(num_symbols):
    """
    生成BPSK调制的信号 (-1或1)
    参数:
        num_symbols (int): 要生成的符号数量
    返回:
        numpy.ndarray: BPSK调制后的符号序列
    """
    bits = np.random.randint(0, 2, num_symbols)  # 生成0或1的随机比特序列
    bpsk_symbols = 2 * bits - 1  # 将0映射到-1，1映射到1
    return bpsk_symbols  # 返回BPSK符号


def pass_through_channel(signal, channel_coeffs):
    """
    使信号通过一个FIR信道 (模拟多径效应和ISI)
    参数:
        signal (numpy.ndarray): 输入信号序列
        channel_coeffs (list or numpy.ndarray): 信道的冲激响应系数
    返回:
        numpy.ndarray: 通过信道后的信号序列
    """
    channel_output_same = np.convolve(signal, channel_coeffs, mode='same')
    return channel_output_same  # 返回通过信道后的信号


def add_noise(signal, snr_db):
    """
    向信号中添加高斯白噪声
    参数:
        signal (numpy.ndarray): 输入信号序列
        snr_db (float): 信噪比 (dB)
    返回:
        numpy.ndarray: 添加噪声后的信号序列
    """
    signal_power = np.mean(np.abs(signal) ** 2)  # 计算信号的平均功率
    snr_linear = 10 ** (snr_db / 10.0)  # 将dB单位的SNR转换为线性单位
    noise_power = signal_power / snr_linear  # 根据SNR计算噪声功率
    noise_std_dev = np.sqrt(noise_power / 2) if np.iscomplexobj(signal) else np.sqrt(noise_power)  # 计算噪声的标准差 (BPSK是实信号)
    noise = np.random.normal(0, noise_std_dev, signal.shape)  # 生成均值为0，标准差为noise_std_dev的高斯噪声
    return signal + noise  # 返回加噪信号


def create_equalizer_input_output(received_signal, original_signal, window_size):
    """
    为均衡器创建输入窗口和对应的期望输出
    参数:
        received_signal (numpy.ndarray): 接收到的信号序列 (已通过信道和噪声)
        original_signal (numpy.ndarray): 原始发送的BPSK符号序列 (用作训练目标)
        window_size (int): 均衡器输入窗口的大小 (应为奇数)
    返回:
        tuple: (X, y)
            X (numpy.ndarray): 均衡器的输入数据 (每个行为一个窗口)
            y (numpy.ndarray): 均衡器的期望输出数据 (每个元素对应一个窗口的中心符号)
    """
    if window_size % 2 == 0:  # 检查窗口大小是否为奇数
        raise ValueError("Window size must be an odd number.")  # 如果不是奇数，则抛出错误

    num_samples = len(received_signal)  # 获取接收信号的长度
    half_window = (window_size - 1) // 2  # 计算半窗口大小

    X_list = []  # 初始化输入数据列表
    y_list = []  # 初始化期望输出数据列表

    for i in range(half_window, num_samples - half_window):  # 遍历接收信号，以创建窗口
        window = received_signal[i - half_window: i + half_window + 1]  # 提取当前窗口的接收信号样本
        X_list.append(window)  # 将窗口添加到输入数据列表
        y_list.append(original_signal[i])  # 提取对应的原始发送符号作为期望输出

    X = np.array(X_list)  # 将输入数据列表转换为NumPy数组
    y = np.array(y_list).reshape(-1, 1)  # 将期望输出数据列表转换为NumPy数组，并调整形状为列向量

    return X, y  # 返回输入数据和期望输出数据


def generate_dataset(num_symbols, channel_coeffs, snr_db, window_size, is_mixed_snr=False, snr_min=0, snr_max=15):
    """
    生成完整的训练或测试数据集
    参数:
        num_symbols (int): 要生成的原始符号数量
        channel_coeffs (list): 信道冲激响应系数
        snr_db (float or None): 信噪比 (dB)。如果 is_mixed_snr=True, 此参数被忽略。
        window_size (int): 均衡器输入窗口大小
        is_mixed_snr (bool): 是否生成混合SNR的数据。若为True, SNR会在snr_min和snr_max之间随机变化。
        snr_min (float): 混合SNR时的最小SNR (dB)
        snr_max (float): 混合SNR时的最大SNR (dB)
    返回:
        tuple: (X, y, original_symbols_aligned, received_signal_aligned)
            X (numpy.ndarray): 均衡器的输入数据
            y (numpy.ndarray): 均衡器的期望输出数据 (原始BPSK符号, -1或1)
            original_symbols_aligned (numpy.ndarray): 与X, y对齐的原始BPSK符号，用于BER计算
            received_signal_aligned (numpy.ndarray): 与X, y对齐的接收信号（仅中心抽头），用于无均衡BER计算
    """

    original_bpsk = generate_bpsk_signal(num_symbols)  # 生成原始BPSK信号
    signal_after_channel = pass_through_channel(original_bpsk, channel_coeffs)  # 信号通过信道

    if is_mixed_snr:  # 如果使用混合SNR
        print(f"生成混合SNR数据，范围 [{snr_min} dB, {snr_max} dB]")  # 打印信息
        # 将信号分成小段，每段应用一个随机SNR
        current_snr_db = np.random.uniform(snr_min, snr_max)  # 在指定范围内随机选择一个SNR值
        # print(f"  当前批次使用 SNR: {current_snr_db:.2f} dB") # 打印当前SNR
        received_noisy_signal = add_noise(signal_after_channel, current_snr_db)  # 添加高斯白噪声
    else:  # 如果不使用混合SNR
        if snr_db is None:  # 检查snr_db是否为None
            raise ValueError("snr_db must be provided if is_mixed_snr is False.")  # 错误
        received_noisy_signal = add_noise(signal_after_channel, snr_db)  # 使用固定的snr_db添加噪声

    X, y = create_equalizer_input_output(received_noisy_signal, original_bpsk, window_size)  # 创建均衡器的输入和输出数据

    original_symbols_aligned_for_ber = y.flatten()  # y是与X对齐的原始符号
    center_tap_index_in_window = (window_size - 1) // 2  # 计算窗口中中心抽头的索引
    received_signal_center_tap_aligned = X[:, center_tap_index_in_window]  # 提取每个窗口中心抽头的接收信号值

    return X, y, original_symbols_aligned_for_ber, received_signal_center_tap_aligned  # 返回数据集


def generate_mixed_snr_training_set(num_symbols_total, channel_coeffs, window_size, snr_min, snr_max, num_snr_steps=10):
    """
    生成一个混合了多种SNR条件的大型训练集。
    数据将在 snr_min 和 snr_max 之间以 num_snr_steps 个离散SNR值生成，并合并。
    参数:
        num_symbols_total (int): 训练集总符号数
        channel_coeffs (list): 信道系数
        window_size (int): 窗口大小
        snr_min (float): 最小SNR (dB)
        snr_max (float): 最大SNR (dB)
        num_snr_steps (int): 在SNR范围内采样的点数
    返回:
        tuple: (X_train_mixed, y_train_mixed)
    """
    print(f"开始生成混合SNR训练集，总符号数: {num_symbols_total}, SNR范围: [{snr_min}dB, {snr_max}dB]...")  # 打印信息

    X_all = []  # 初始化总输入数据列表
    y_all = []  # 初始化总期望输出数据列表

    symbols_per_snr_step = num_symbols_total // num_snr_steps  # 计算每个SNR步骤的符号数

    snr_values_for_training = np.linspace(snr_min, snr_max, num_snr_steps)  # 生成SNR值序列

    for snr_val in tqdm(snr_values_for_training, desc="Generating mixed SNR data", unit="SNR_step"):  # 遍历每个SNR值
        X_s, y_s, _, _ = generate_dataset(
            symbols_per_snr_step,
            channel_coeffs,
            snr_val,  # 当前SNR值
            window_size,
            is_mixed_snr=False  # 我们是在这个循环的更高层级控制混合
        )  # 生成当前SNR值下的数据
        X_all.append(X_s)  # 添加到总输入数据列表
        y_all.append(y_s)  # 添加到总期望输出数据列表

    X_train_mixed = np.concatenate(X_all, axis=0)  # 合并所有输入数据
    y_train_mixed = np.concatenate(y_all, axis=0)  # 合并所有期望输出数据

    # 打乱整个混合数据集
    permutation = np.random.permutation(X_train_mixed.shape[0])  # 生成随机排列索引
    X_train_mixed = X_train_mixed[permutation]  # 打乱输入数据
    y_train_mixed = y_train_mixed[permutation]  # 打乱期望输出数据

    print(f"混合SNR训练集生成完毕。总样本数: {X_train_mixed.shape[0]}")  # 打印完成信息
    return X_train_mixed, y_train_mixed  # 返回混合训练集


