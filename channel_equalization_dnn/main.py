import os
import time
import numpy as np
import config
import data_generator
from metrics import calculate_ber, plot_ber_comparison
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler  # 用于输入归一化


DNN_IMPLEMENTATION_CHOICE = ''  # 初始化选择变量

def run_simulation():  # 定义主仿真函数
    global DNN_IMPLEMENTATION_CHOICE  # 声明使用全局变量

    # -------------------- 选择DNN实现 --------------------
    while DNN_IMPLEMENTATION_CHOICE.lower() not in ['tf', 'numpy']:
        print("\n请选择DNN实现方式:")  # 提示选择
        print("1. TensorFlow (tf)")  # 选项1
        print("2. NumPy (numpy)")  # 选项2
        choice = input("请输入选择 (tf/numpy): ").strip().lower()  # 获取输入并处理
        if choice == 'tf' or choice == '1':  # 如果选择TensorFlow
            DNN_IMPLEMENTATION_CHOICE = 'tf'  # 设置选择为'tf'
        elif choice == 'numpy' or choice == '2':  # 如果选择NumPy
            DNN_IMPLEMENTATION_CHOICE = 'numpy'  # 设置选择为'numpy'
        else:
            print("无效输入，请重新选择。")  # 提示无效输入

    if DNN_IMPLEMENTATION_CHOICE == 'tf':  # 如果选择TensorFlow
        from dnn_tf_equalizer import DNN_TF_Equalizer as DNN_Equalizer  # 导入TensorFlow版DNN均衡器
        print("已选择 TensorFlow 实现的DNN。")  # 打印选择信息
    else:  # DNN_IMPLEMENTATION_CHOICE == 'numpy'
        from dnn_numpy_equalizer import DNN_Numpy_Equalizer as DNN_Equalizer  # 导入NumPy版DNN均衡器
        print("已选择 NumPy 手动实现的DNN。")  # 打印选择信息

    from lms_equalizer import LMS_Equalizer  # 导入LMS均衡器模块

    # -------------------- 初始化 --------------------
    start_total_time = time.time()  # 记录程序开始运行时间

    # 从config加载参数
    snr_db_range = config.SNR_DB_RANGE
    num_symbols_train_total = config.NUM_SYMBOLS_TRAIN  # 总训练符号数
    num_symbols_test = config.NUM_SYMBOLS_TEST
    channel_coeffs = config.CHANNEL_COEFFS
    window_size = config.WINDOW_SIZE

    # DNN参数
    dnn_hidden_sizes = config.DNN_HIDDEN_SIZES
    dnn_dropout_rate = config.DNN_DROPOUT_RATE  # Dropout率
    dnn_epochs = config.DNN_EPOCHS
    dnn_batch_size = config.DNN_BATCH_SIZE
    dnn_initial_lr = config.DNN_LEARNING_RATE  # 初始学习率
    lr_decay_epochs = config.LR_DECAY_EPOCHS
    lr_decay_factor = config.LR_DECAY_FACTOR
    early_stopping_patience = config.EARLY_STOPPING_PATIENCE
    min_delta = config.MIN_DELTA
    normalize_input = config.NORMALIZE_INPUT  # 是否归一化输入

    # LMS参数
    lms_taps = config.LMS_NUM_TAPS
    lms_mu = config.LMS_MU
    lms_epochs = config.LMS_EPOCHS

    ber_results = {  # 初始化BER结果字典
        "No Equalization": [],
        "LMS": [],
        f"DNN ({DNN_IMPLEMENTATION_CHOICE.upper()})": []
    }

    # --- 全局随机种子 ---
    np.random.seed(config.RANDOM_SEED)  # 设置NumPy随机种子
    # TensorFlow的种子在其类初始化时基于config.RANDOM_SEED设置

    # --- 生成统一的、混合SNR的训练/验证集 (如果启用) ---
    X_train_full, y_train_full = None, None
    scaler = None  # 初始化归一化器

    if config.USE_MIXED_SNR_TRAINING:  # 如果使用混合SNR训练
        print("\n--- 生成SNR训练/验证集 ---")  # 打印信息
        # 使用 data_generator 中的新函数
        X_train_full, y_train_full = data_generator.generate_mixed_snr_training_set(
            num_symbols_total=num_symbols_train_total,
            channel_coeffs=channel_coeffs,
            window_size=window_size,
            snr_min=config.TRAIN_SNR_DB_MIN,
            snr_max=config.TRAIN_SNR_DB_MAX,
            num_snr_steps=10  # 例如，混合10个不同SNR级别的数据
        )  # 生成混合SNR训练集
    else:  # 如果不使用混合SNR训练 (按每个SNR点分别训练)
        print("\n--- 注意: DNN将针对每个SNR点分别训练 (USE_MIXED_SNR_TRAINING=False) ---")  # 打印提示信息
        # 在这种模式下，训练数据将在SNR循环内部生成

    if X_train_full is not None and normalize_input:  # 如果已生成训练集且需要归一化
        print("对训练数据进行归一化...")  # 打印归一化信息
        scaler = StandardScaler()  # 实例化StandardScaler
        X_train_full = scaler.fit_transform(X_train_full)  # 拟合并转换训练数据
        print("归一化完成。Scaler已在训练数据上拟合。")  # 打印完成信息

    # 将完整训练集划分为训练子集和验证子集 (7:3)
    X_train_dnn, y_train_dnn = None, None  # DNN训练用
    X_val_dnn, y_val_dnn = None, None  # DNN验证用

    if X_train_full is not None:  # 如果已生成训练集
        val_split_idx = int(0.7 * X_train_full.shape[0])  # 计算验证集分割点
        X_train_dnn = X_train_full[:val_split_idx]  # 分割训练输入
        y_train_dnn = y_train_full[:val_split_idx]  # 分割训练目标
        X_val_dnn = X_train_full[val_split_idx:]  # 分割验证输入
        y_val_dnn = y_train_full[val_split_idx:]  # 分割验证目标
        print(f"训练集划分为: 训练 {X_train_dnn.shape[0]} 样本, 验证 {X_val_dnn.shape[0]} 样本")  # 打印划分信息

    # --- 初始化DNN和LMS模型实例 ---
    # DNN模型只需要初始化一次，如果使用混合SNR训练数据
    dnn_equalizer = None
    if config.USE_MIXED_SNR_TRAINING:  # 如果使用混合SNR训练
        print(f"\n--- 初始化和训练 DNN ({DNN_IMPLEMENTATION_CHOICE.upper()}) 使用SNR数据 ---")  # 打印信息
        dnn_equalizer = DNN_Equalizer(input_size=window_size,
                                      hidden_sizes=dnn_hidden_sizes,
                                      output_size=1,
                                      dropout_rate=dnn_dropout_rate,  # 传递dropout率
                                      random_seed=config.RANDOM_SEED)  # 实例化DNN均衡器

        dnn_equalizer.train(X_train_dnn, y_train_dnn, X_val_dnn, y_val_dnn,
                            epochs=dnn_epochs,
                            batch_size=dnn_batch_size,
                            initial_learning_rate=dnn_initial_lr,
                            lr_decay_epochs=lr_decay_epochs,
                            lr_decay_factor=lr_decay_factor,
                            early_stopping_patience=early_stopping_patience,
                            min_delta=min_delta)  # 训练DNN均衡器
        print(f"DNN ({DNN_IMPLEMENTATION_CHOICE.upper()}) 使用SNR数据训练完成。")  # 打印完成信息

    # -------------------- 主循环：遍历不同SNR进行测试 --------------------
    snr_pbar = tqdm(snr_db_range, desc="Overall SNR Progress", unit="SNR_dB")  # 创建SNR进度条
    for snr_db in snr_pbar:  # 遍历每个SNR值
        snr_pbar.set_postfix({"Current_SNR": f"{snr_db} dB"})  # 更新进度条
        print(f"\n--- 开始处理 SNR = {snr_db} dB ---")  # 打印当前SNR处理信息

        print("正在生成测试数据...")  # 打印生成测试数据信息
        X_test_raw, _, y_test_original_aligned, received_test_center_tap = data_generator.generate_dataset(
            num_symbols_test, channel_coeffs, snr_db, window_size,
            is_mixed_snr=False  # 测试数据针对特定SNR
        )  # 生成测试数据集

        X_test_lms = X_test_raw  # LMS使用原始测试数据
        X_test_dnn_eval = X_test_raw  # DNN评估也使用原始测试数据

        if normalize_input:  # 如果需要归一化
            if scaler is None and not config.USE_MIXED_SNR_TRAINING:  # 如果没有全局scaler且不是混合训练
                # 需要为当前SNR的训练数据拟合一个scaler
                print("为当前SNR的DNN训练数据拟合归一化器...")  # 打印信息
                _, temp_y_train_for_scaler, _, _ = data_generator.generate_dataset(
                    num_symbols_train_total // len(snr_db_range),  # 分配一部分训练符号
                    channel_coeffs, snr_db, window_size, is_mixed_snr=False
                )
                temp_scaler = StandardScaler()  # 实例化StandardScaler
                X_temp_train_for_scaler = temp_y_train_for_scaler
                X_train_current_snr, y_train_current_snr, _, _ = data_generator.generate_dataset(
                    num_symbols_train_total // len(snr_db_range),  # 分配一部分训练符号
                    channel_coeffs, snr_db, window_size, is_mixed_snr=False
                )
                temp_scaler.fit(X_train_current_snr)  # 拟合归一化器
                X_test_dnn_eval = temp_scaler.transform(X_test_raw)  # 转换测试数据
                print("测试数据已使用当前SNR训练数据拟合的Scaler归一化。")  # 打印完成信息
            elif scaler:  # 如果有全局scaler (来自混合训练)
                X_test_dnn_eval = scaler.transform(X_test_raw)  # 使用全局scaler转换测试数据

        print(f"测试数据: {X_test_dnn_eval.shape[0]}样本")  # 打印测试数据量

        # 无均衡器性能
        ber_no_eq = calculate_ber(y_test_original_aligned, received_test_center_tap)  # 计算无均衡BER
        ber_results["No Equalization"].append(ber_no_eq)  # 添加到结果列表
        print(f"SNR={snr_db}dB, 无均衡BER: {ber_no_eq:.6f}")  # 打印无均衡BER

        # LMS均衡器 (总是在当前SNR的训练数据上训练)
        print("\n--- LMS均衡器 ---")  # 打印LMS均衡器处理信息
        # 为LMS生成当前SNR的训练数据
        X_train_lms, y_train_lms, _, _ = data_generator.generate_dataset(
            num_symbols_train_total // len(
                snr_db_range) if not config.USE_MIXED_SNR_TRAINING else num_symbols_train_total // 10,  # LMS训练数据量
            channel_coeffs, snr_db, window_size, is_mixed_snr=False
        )  # 生成LMS训练数据
        lms_equalizer = LMS_Equalizer(num_taps=lms_taps, mu=lms_mu, random_seed=config.RANDOM_SEED)  # 实例化LMS均衡器
        lms_equalizer.train(X_train_lms, y_train_lms, epochs=lms_epochs)  # 训练LMS均衡器
        y_pred_lms = lms_equalizer.predict(X_test_lms)  # 使用LMS进行预测
        ber_lms = calculate_ber(y_test_original_aligned, y_pred_lms)  # 计算LMS的BER
        ber_results["LMS"].append(ber_lms)  # 添加到结果列表
        print(f"SNR={snr_db}dB, LMS BER: {ber_lms:.6f}")  # 打印LMS的BER

        # DNN均衡器
        if not config.USE_MIXED_SNR_TRAINING:  # 如果DNN是按每个SNR点分别训练
            print(f"\n--- DNN ({DNN_IMPLEMENTATION_CHOICE.upper()})均衡器 (为SNR={snr_db}dB单独训练) ---")  # 打印DNN处理信息
            # 生成当前SNR的DNN训练/验证数据
            X_train_curr_snr_full, y_train_curr_snr_full, _, _ = data_generator.generate_dataset(
                num_symbols_train_total,  # 使用配置的总训练符号数
                channel_coeffs, snr_db, window_size, is_mixed_snr=False
            )  # 生成当前SNR的DNN训练数据

            if normalize_input:  # 如果需要归一化
                current_snr_scaler = StandardScaler()  # 实例化StandardScaler
                X_train_curr_snr_full = current_snr_scaler.fit_transform(X_train_curr_snr_full)  # 拟合并转换

            val_split_idx_curr = int(0.7 * X_train_curr_snr_full.shape[0])  # 计算验证集分割点
            X_train_dnn_curr = X_train_curr_snr_full[:val_split_idx_curr]  # 分割训练输入
            y_train_dnn_curr = y_train_curr_snr_full[:val_split_idx_curr]  # 分割训练目标
            X_val_dnn_curr = X_train_curr_snr_full[val_split_idx_curr:]  # 分割验证输入
            y_val_dnn_curr = y_train_curr_snr_full[val_split_idx_curr:]  # 分割验证目标

            dnn_equalizer_current_snr = DNN_Equalizer(input_size=window_size,
                                                      hidden_sizes=dnn_hidden_sizes,
                                                      output_size=1,
                                                      dropout_rate=dnn_dropout_rate,
                                                      random_seed=config.RANDOM_SEED)  # 实例化DNN均衡器
            dnn_equalizer_current_snr.train(X_train_dnn_curr, y_train_dnn_curr, X_val_dnn_curr, y_val_dnn_curr,
                                            epochs=dnn_epochs, batch_size=dnn_batch_size,
                                            initial_learning_rate=dnn_initial_lr,
                                            lr_decay_epochs=lr_decay_epochs, lr_decay_factor=lr_decay_factor,
                                            early_stopping_patience=early_stopping_patience,
                                            min_delta=min_delta)  # 训练DNN

            y_pred_dnn = dnn_equalizer_current_snr.predict(X_test_dnn_eval)  # 使用当前SNR训练的DNN预测
            if DNN_IMPLEMENTATION_CHOICE == 'tf' and hasattr(dnn_equalizer_current_snr, 'close_session'):  # 如果是TF
                dnn_equalizer_current_snr.close_session()  # 关闭会话
        else:
            print(f"\n--- DNN ({DNN_IMPLEMENTATION_CHOICE.upper()})均衡器 (使用已训练的SNR模型) ---")  # 打印信息
            if dnn_equalizer is None:  # 检查模型是否存在
                raise RuntimeError("SNR模型未被训练！请检查USE_MIXED_SNR_TRAINING设置。")  # 错误
            y_pred_dnn = dnn_equalizer.predict(X_test_dnn_eval)  # 使用预训练的混合SNR模型预测

        ber_dnn = calculate_ber(y_test_original_aligned, y_pred_dnn)  # 计算DNN的BER
        ber_results[f"DNN ({DNN_IMPLEMENTATION_CHOICE.upper()})"].append(ber_dnn)  # 添加到结果列表
        print(f"SNR={snr_db}dB, DNN ({DNN_IMPLEMENTATION_CHOICE.upper()}) BER: {ber_dnn:.6f}")  # 打印DNN的BER

    # 如果DNN是混合SNR训练的，在所有SNR测试完成后关闭会话 (TF)
    if config.USE_MIXED_SNR_TRAINING and DNN_IMPLEMENTATION_CHOICE == 'tf' and dnn_equalizer and hasattr(dnn_equalizer,
                                                                                                         'close_session'):
        dnn_equalizer.close_session()  # 关闭TensorFlow会话

    # -------------------- 结果汇总与绘图 --------------------
    end_total_time = time.time()  # 记录结束时间
    total_runtime = end_total_time - start_total_time  # 计算总运行时间

    print("\n\n--- 仿真完成 ---")  # 打印完成信息
    print(f"程序总运行时间: {total_runtime:.2f} 秒 ({total_runtime / 60:.2f} 分钟)")  # 打印总运行时间

    print("\n--- BER 结果汇总 ---")  # 打印BER结果汇总
    for snr_idx, snr_val in enumerate(snr_db_range):  # 遍历SNR值
        print(f"SNR = {snr_val} dB:")  # 打印当前SNR
        for algo_name, bers in ber_results.items():  # 遍历各算法BER
            if snr_idx < len(bers):  # 检查索引有效性
                print(f"  {algo_name}: {bers[snr_idx]:.6f}")  # 打印BER
            else:
                print(f"  {algo_name}: 数据缺失")  # 打印数据缺失信息

    plot_ber_comparison(snr_db_range, ber_results,
                        title=f"BPSK LMS vs (DNN: {DNN_IMPLEMENTATION_CHOICE.upper()})")  # 调用绘图函数


if __name__ == '__main__':  # 如果该脚本作为主程序运行
    # 确保TensorFlow 1.x的 Saver 使用的目录存在
    if not os.path.exists("./tf_best_model.ckpt.index"):  # 简单检查，或直接创建目录
        # os.makedirs("./tf_models", exist_ok=True) # 如果模型保存在子目录
        pass
    run_simulation()  # 调用主仿真函数

