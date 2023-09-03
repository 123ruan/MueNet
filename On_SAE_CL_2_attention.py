import scipy.io as scio
import numpy as np
from sklearn import preprocessing
from scipy import stats
import torch
from model_attention_Mu_GMM import Multi_SAE
from time import time

# 定义函数
def cal_threshold(x, alpha):
    kernel = stats.gaussian_kde(x)
    step = np.linspace(0, 100, 10000)
    pdf = kernel(step)
    for i in range(len(step)):
        if sum(pdf[0:(i + 1)]) / sum(pdf) > alpha:
            break
    return step[i + 1]


# 计算检测性能
def cal_FR(statistics, limit):
    mn = 0
    FR = 0
    for i in range(len(statistics)):
        if statistics[i] > limit[0]:
            mn = mn + 1
        FR = mn / len(statistics)
    return FR


def get_index1(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]


def moving_average(l, N):
    sum = 0
    result = list(0 for x in l)
    for i in range(0, N):
        sum = sum + l[i]
        result[i] = sum / (i + 1)
    for i in range(N, len(l)):
        sum = sum - l[i - N] + l[i]
        result[i] = sum / N
    return np.array(result)


# 载入数据
start = time()
n_th = 160
alpha = 0.95
data0 = scio.loadmat('F:\合作论文\MuSAE\data/te_data.mat')  # data为字典类型
data1 = scio.loadmat('F:\合作论文\MuSAE\data/te_training_data.mat')  # data为字典类型
data_training = np.vstack((data0['d00_te'], data1['d00']))
data_training = np.hstack((data_training[:, 0:22], data_training[:, 41:52]))  # 提取33个变量

scaler = preprocessing.StandardScaler().fit(data_training)  # 创建标准化转换器
X_training = preprocessing.scale(data_training)  # 标准化处理
data_w = []
#
#
list_1 = [1]
for i in list_1:
    if i <= 9:
        name = "d0{}_te".format(i)
        # print(name)
    else:
        name = "d{}_te".format(i)
    print("--------------{}-----------".format(name))
    data_test = data0[name]  # 载入测试数据
    data_test = np.hstack((data_test[:, 0:22], data_test[:, 41:52]))  # 提取33个变量
    X_test = scaler.transform(data_test)  # 测试数据标准化
    n_test = X_test.shape[0]
    m_test = X_test.shape[1]

    #GMM
    index1 = [0, 3, 4, 5, 9, 10, 14, 17, 18, 24, 25, 27, 29, 30]
    index2 = [1, 8, 21, 22, 31, 32]
    index3 = [2, 6, 7, 11, 12, 13, 15, 16, 19, 20, 23, 26, 28]

    X_1 = X_test[:, index1]
    X_2 = X_test[:, index2]
    X_3 = X_test[:, index3]
    X_test = np.hstack((X_1, X_2, X_3))

    autoencoder_1 = Multi_SAE(len(index1), len(index2), len(index3))

    autoencoder_1.load_state_dict(torch.load("F:\合作论文\Ablation\TE\Batchsize/2-1checkpoint.pt"))  # 之前一直使用

    X_1 = torch.from_numpy(X_1)
    X_1 = X_1.to(torch.float32)
    X_2 = torch.from_numpy(X_2)
    X_2 = X_2.to(torch.float32)
    X_3 = torch.from_numpy(X_3)
    X_3 = X_3.to(torch.float32)

    autoencoder_1.eval()
    output, feature1, feature2, feature3, output1, output2, output3 = autoencoder_1(X_1, X_2, X_3)

    feature = torch.cat((feature1, feature2, feature3), 1)
    X_test_reconstruct = torch.cat((output1, output2, output3), 1)
    feature = feature.detach().numpy()
    X_test_reconstruct = X_test_reconstruct.detach().numpy()
    print(X_test_reconstruct.shape)


    # 构造一个统计量 ----------------正常来说用这个
    T2_1 = np.ones(n_test)
    for i in range(n_test):
        a = feature[i, :]
        T2_1[i] = np.dot(a, a.T)
    T2_1 = moving_average(T2_1, 5)
    th_T2_1 = cal_threshold(T2_1[0:n_th], alpha)
    th_T2_1 = th_T2_1 * np.ones(n_test)

    # 构造Q统计量
    Q_1 = np.ones(n_test)
    Q0_1 = X_test - X_test_reconstruct
    for i in range(0, 960):
        Q_1[i] = np.dot(Q0_1[i, :], Q0_1[i, :].T)
    Q_1 = moving_average(Q_1, 5)
    th_Q_1 = cal_threshold(Q_1[0:n_th], alpha)
    th_Q_1 = th_Q_1 * np.ones(n_test)

    # 计算性能
    FAR_T2_1 = cal_FR(T2_1[0:n_th], th_T2_1)
    print("T-A-1:The false alarm rate of SAE T² is: " + str(FAR_T2_1))
    FDR_T2_1 = cal_FR(T2_1[n_th:n_test], th_T2_1)
    print("T-D-1:The detection rate of SAE T² is: " + str(FDR_T2_1))

    print("------------------")

    FAR_Q_1 = cal_FR(Q_1[0:n_th], th_Q_1)
    print("Q-A-1:The false alarm rate of SAE Q is: " + str(FAR_Q_1))
    FDR_Q_1 = cal_FR(Q_1[n_th:n_test], th_Q_1)
    print("Q-D-1:The detection rate of SAE Q is: " + str(FDR_Q_1))

    data_performance = [FDR_T2_1, FAR_T2_1, FDR_Q_1, FAR_Q_1]
    data_w.append(data_performance)

    end = time()
    run_time = end - start
    print("block_test_time is {:.2f}s".format(run_time))
    # scipy.io.savemat("./data_mat/Ablation/Batchsize-25/TE_Ours_data{}.mat".format(name), {"data_{}".format(name): Q_1})
    # scipy.io.savemat("./data_mat/Ablation/Batchsize-25/E_Ours_label{}.mat".format(name), {"limit_{}".format(name): th_Q_1})
