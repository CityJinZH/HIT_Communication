import numpy as np  # 导入NumPy库
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
from tqdm import tqdm  # 导入tqdm库，用于显示进度条


def calculate_ber(y_true, y_pred_continuous):
    """
    计算误码率 (Bit Error Rate - BER)
    参数:
        y_true (numpy.ndarray): 真实的BPSK符号序列 (-1或1), shape (num_samples,) or (num_samples, 1)
        y_pred_continuous (numpy.ndarray): 均衡器输出的连续值估计, shape (num_samples,) or (num_samples, 1)
    返回:
        float: 计算得到的误码率
    """
    # 确保y_true是一维的
    y_true_flat = y_true.flatten()  # 将真实符号展平为一维数组

    # 确保y_pred_continuous是一维的
    y_pred_continuous_flat = y_pred_continuous.flatten()  # 将预测的连续值展平为一维数组

    # 对连续预测值进行判决得到BPSK符号
    # 大于0判为+1, 小于等于0判为-1 (sign)
    y_pred_symbols = np.sign(y_pred_continuous_flat)  # 使用sign函数进行硬判决
    y_pred_symbols[y_pred_symbols == 0] = 1  # 将判决结果为0的符号映射为1

    # 计算错误比特的数量
    num_errors = np.sum(y_true_flat != y_pred_symbols)  # 比较真实符号和判决符号，计算不同符号的数量

    # 计算总比特数
    total_bits = len(y_true_flat)  # 获取总比特数

    # 计算BER
    ber = num_errors / total_bits if total_bits > 0 else 0.0  # 计算误码率，避免除以零

    return ber  # 返回误码率


def plot_ber_comparison(snr_range, ber_results, title="BER Performance Comparison"):
    """
    绘制不同算法的BER vs SNR曲线图
    参数:
        snr_range (list or numpy.ndarray): SNR值列表 (dB)
        ber_results (dict): 字典，键是算法名称 (str)，值是对应的BER列表 (list of float)
        title (str): 图表的标题
    """
    plt.figure(figsize=(10, 7))  # 创建一个新的图形窗口，并设置大小

    markers = ['o', 's', '^', 'd', 'x', '*', '+']  # 定义不同曲线的标记样式
    marker_idx = 0  # 初始化标记索引

    for algorithm_name, bers in ber_results.items():  # 遍历ber_results字典中的每个算法及其BER数据
        if len(bers) == len(snr_range):  # 检查BER数据长度是否与SNR范围长度一致
            plt.plot(snr_range, bers, marker=markers[marker_idx % len(markers)], linestyle='-',
                     label=algorithm_name)  # 绘制BER曲线
            marker_idx += 1  # 更新标记索引
        else:
            print(
                f"警告: 算法 '{algorithm_name}' 的BER数据点数量 ({len(bers)}) 与SNR范围 ({len(snr_range)}) 不匹配。跳过绘制。")  # 打印警告信息

    plt.yscale('log')  # 将y轴设置为对数刻度，因为BER通常跨越多个数量级
    plt.xlabel('SNR (dB)')  # 设置x轴标签
    plt.ylabel('Bit Error Rate (BER)')  # 设置y轴标签
    plt.title(title)  # 设置图表标题
    plt.grid(True, which="both", ls="--")  # 添加网格线，'both'表示主次刻度都显示，'ls'设置线型
    plt.legend()  # 显示图例
    plt.ylim(bottom=1e-5, top=1.0)  # 设置y轴的显示范围，确保低BER值可见，最高为1.0
    # plt.xlim(left=min(snr_range)-1, right=max(snr_range)+1) # 可选：设置x轴范围

    plt.show()  # 显示图表
    print("BER比较图已生成。")  # 打印图表生成信息

