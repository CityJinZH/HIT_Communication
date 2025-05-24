# 数据生成参数
NUM_SYMBOLS_TRAIN = 1000000     # 训练符号数量
NUM_SYMBOLS_TEST = 1000000     # 测试符号，以便更准确测量低BER
CHANNEL_COEFFS = [0.5, 1.0, -0.3, 0.2]
WINDOW_SIZE = 9                 # 窗口大小, 7, 9, 11
SNR_DB_RANGE = list(range(0, 14, 2))

TRAIN_SNR_DB_MIN = 2
TRAIN_SNR_DB_MAX = 12
USE_MIXED_SNR_TRAINING = True   # 是否使用混合SNR数据进行训练


# DNN 训练参数
DNN_EPOCHS = 100                # 训练轮数，配合早停使用
DNN_BATCH_SIZE = 256
DNN_LEARNING_RATE = 0.00025       # 初始学习率


# DNN 结构参数
DNN_HIDDEN_SIZES = [16, 24, 32, 24] # 更深/更宽的网络
DNN_DROPOUT_RATE = 0.1          # Dropout比率 (0表示不使用)


# 早停参数
EARLY_STOPPING_PATIENCE = 10    # 验证损失连续多少轮不下降则停止
MIN_DELTA = 0.0001              # 认为损失下降的最小阈值


# 学习率衰减参数 (每N轮衰减)
LR_DECAY_EPOCHS = 20            # 每多少轮衰减一次学习率
LR_DECAY_FACTOR = 0.3           # 学习率衰减因子


# LMS 参数
LMS_NUM_TAPS = WINDOW_SIZE     # 保持与DNN窗口大小一致
LMS_MU = 0.025                   # LMS步长
LMS_EPOCHS = 1


# 随机种子
RANDOM_SEED = 42


# 是否进行输入归一化
NORMALIZE_INPUT = True
