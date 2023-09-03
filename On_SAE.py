# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:33:58 2019

@author: Administrator
"""
from time import time
import scipy
import scipy.io as scio
import scipy.io as scio
import numpy as np 
# from keras.models import load_model
import torch
from sklearn import preprocessing
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics as ms
import pandas as pd
import torch.nn as nn
import openpyxl

# 定义函数
def cal_threshold(x, alpha):
    kernel = stats.gaussian_kde(x)
    step = np.linspace(0,100,10000)
    pdf = kernel(step)
    for i in range(len(step)):
        if sum(pdf[0:(i+1)]) / sum(pdf) > alpha:    
            break
    return step[i+1]


# 计算检测性能
def cal_FR(statistics, limit):
    mn = 0
    FR = 0
    for i in range(len(statistics)):
        if statistics[i] > limit[0]:
            mn = mn+1
        FR = mn/len(statistics)
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
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(33, 50)
        # self.encoder = nn.Sequential(
        #     nn.Linear(33, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 30),
        #     nn.Tanh(),
        #     nn.Linear(30, 25),
        #     nn.Tanh())
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(25, 30),
        #     nn.Tanh(),
        #     nn.Linear(30, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 33))
        # self.encoder = nn.Sequential(  #对应结构50-33-18-33-50,9-4-18-4-9,15-6-18-6-15
        #     nn.Linear(33, 95),
        #     nn.Tanh(),
        #     nn.Linear(95, 40),
        #     nn.Tanh(),
        #     nn.Linear(40, 18),
        #     nn.Tanh())
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(18, 40),
        #     nn.Tanh(),
        #     nn.Linear(40, 95),
        #     nn.Tanh(),
        #     nn.Linear(95, 33))
        # self.encoder = nn.Sequential(  # 用于MuSAE的消融实验，共292个神经元。
        #     nn.Linear(33, 98),
        #     nn.Tanh(),
        #     nn.Linear(98, 40),
        #     nn.Tanh(),
        #     nn.Linear(40, 16),
        #     nn.Tanh())
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(16, 40),
        #     nn.Tanh(),
        #     nn.Linear(40, 98),
        #     nn.Tanh(),
        #     nn.Linear(98, 33))
        # self.encoder = nn.Sequential(  # 用于中文专刊的消融实验，共176个神经元。
        #     nn.Linear(33, 106),
        #     nn.Tanh(),
        #     nn.Linear(106, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 20),
        #     nn.Tanh())
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(20, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 106),
        #     nn.Tanh(),
        #     nn.Linear(106, 33))
        # self.encoder = nn.Sequential(  # 对应结构50-33-18-33-50,9-4-18-4-9,15-6-18-6-15
        #     nn.Linear(33, 76),
        #     nn.Tanh(),
        #     nn.Linear(76, 60),
        #     nn.Tanh(),
        #     nn.Linear(60, 27),
        #     nn.Tanh())
        #
        # self.decoder = nn.Sequential(
        #     nn.Linear(27, 60),
        #     nn.Tanh(),
        #     nn.Linear(60, 76),
        #     nn.Tanh(),
        #     nn.Linear(76, 33))
        self.encoder = nn.Sequential(  # 对应结构50-33-18-33-50,9-4-18-4-9,15-6-18-6-15
            nn.Linear(33, 90),
            nn.Tanh(),
            nn.Linear(90, 56),
            nn.Tanh(),
            nn.Linear(56, 10),
            nn.Tanh())

        self.decoder = nn.Sequential(
            nn.Linear(10, 56),
            nn.Tanh(),
            nn.Linear(56, 90),
            nn.Tanh(),
            nn.Linear(90, 33))

    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.view(batch_size,-1)
        x_encoded = self.encoder(x)
        x_recon = self.decoder(x_encoded)
        return x_encoded, x_recon

start = time()
# 载入数据
n_th =160
alpha = 0.95
perf_T2 = np.zeros([2, 15])
perf_Q = np.zeros([2, 15]) # 记录各个尺度的检测性能
data0 = scio.loadmat('F:\合作论文\MuSAE\data/te_data.mat')# data为字典类型
data1 = scio.loadmat('F:\合作论文\MuSAE\data/te_training_data.mat')# data为字典类型
data_training0 = data0['d00_te']# 访问字典中的一部分
data_training1 = data1['d00']
data_training=np.vstack((data_training0,data_training1))

data_training_all = np.hstack((data_training[:,0:22],data_training[:,41:52]))# 提取33个变量
scaler = preprocessing.StandardScaler().fit(data_training_all)# 创建标准化转换器
X_training = preprocessing.scale(data_training_all)# 标准化处理
n_training = X_training.shape[0]
m_training = X_training.shape[1]
data_w = []

for k in range(1,11):
    autoencoder = Net()
    # autoencoder.load_state_dict(torch.load("./性能汇总/消融实验/纯SAE/9-1-one_model_05checkpoint.pt"))
    autoencoder.load_state_dict(torch.load("F:\合作论文\Ablation\TE\稳定性实验\SAE/{}-checkpoint.pt".format(k)))
    print("----------------------------加载第{}个模型----------------------".format(k))
    for i in range(1,22):
        if i<= 9:
            name = "d0{}_te".format(i)
            # print(name)
        else:
            name = "d{}_te".format(i)
        print("--------------{}-----------".format(name))
        data_test = data0[name]# 载入测试数据


        data_test1 = np.hstack((data_test[:,0:22],data_test[:,41:52]))# 提取33个变量
        data_testing = scaler.transform(data_test1)# 测试数据标准化
        X_test = data_testing
        n_test = X_test.shape[0]
        m_test = X_test.shape[1]

        # SAE = load_model('SAE.h5')
        # SAE_encoder = load_model('SAE_encoder.h5')
        #gai l

        X_test = torch.from_numpy(X_test)
        X_test = X_test.to(torch.float32)

        feature,X_test_reconstruct = autoencoder(X_test)

        feature = feature.detach().numpy()
        X_test_reconstruct = X_test_reconstruct.detach().numpy()
        # 构造T2统计量
        # feature = SAE_encoder.predict(X_test)     #gai l
        T2 = np.ones(n_test)
        for i in range(n_test):
            a = feature[i, :]
            T2[i] = np.dot(a, a.T)

        #T2_1 = moving_average(T2_1, 5)

        th_T2 = cal_threshold(T2[0:n_th], alpha)
        th_T2 = th_T2*np.ones(n_test)
        # 构造Q统计量
        Q = np.ones(n_test)
        # X_test_reconstruct=SAE.predict(X_test)     #gai l
        Q0=X_test-X_test_reconstruct
        for i in range(0,960):
            t1=Q0[i,:].reshape((len(Q0[i,:]),1))
            t2=Q0[i,:].reshape((1,len(Q0[i,:])))
            Q[i] = np.dot(t2,t1)

        Q = moving_average(Q, 5)
        th_Q = cal_threshold(Q[0:n_th], alpha)
        #Q_1 = Q_1/m_test
        #th_Q_1 = th_Q_1/m_test
        th_Q = th_Q*np.ones(n_test)


        #计算性能
        FAR_T2 = cal_FR(T2[0:n_th], th_T2)
        print ("The false alarm rate of SAE T² is: " + str(FAR_T2))
        FDR_T2 = cal_FR(T2[n_th:n_test], th_T2)
        print ("The detection rate of SAE T² is: " + str(FDR_T2))
        FAR_Q = cal_FR(Q[0:n_th], th_Q)
        print ("The false alarm rate of SAE Q is: " + str(FAR_Q))
        FDR_Q = cal_FR(Q[n_th:n_test], th_Q)
        print ("The detection rate of SAE Q is: " + str(FDR_Q))
        data_performance = [FDR_T2, FAR_T2, FDR_Q, FAR_Q]
        data_w.append(data_performance)
        end = time()
        run_time = end - start
        # print(run_time)

# 创建一个新的Excel工作簿
    wb = openpyxl.Workbook()

    # 选择要操作的工作表（默认创建的工作表为第一个）
    ws = wb.active
    # 向工作表写入数据
    ws.append(["DR-T2","FDR-T2","DR-Q","FDR-Q"])
    for row in data_w:
        ws.append(row)

    # 保存工作簿到文件
    wb.save('example_{}.xlsx'.format(k))

    # 关闭工作簿
    wb.close()

# print("block_test_time is {:.2f}s".format(run_time))
#         # 画图T2
#         #plt.figure(1)
#         plt.figure(figsize=(7,5))
#         plt.subplot(211)
#         plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                         wspace=None, hspace=0.4)
#         plt.plot(T2,'b',lw=1.5,label='mornitoring index:T²')#参数控制颜色和字体的粗细
#         plt.plot(th_T2,'r',label='control limit')
#         #plt.plot(d2, 'ro')#每个样本点加红点
#         #plt.plot(d2.cumsum(),'r',lw=1.5)
#         plt.grid(True)
#         #plt.axis('tight')
#         #plt.ylim(np.min(d2.cumsum())- 0.1, np.max(d2.cumsum()) + 0.1)
#         #plt.plot(d2.cumsum(),'ro')
#         plt.legend(loc = 0,fontsize = 14)
#         plt.title('Mornitoring performance of SAE')
#         plt.xlabel('Sample number',fontsize = 14)
#         plt.ylabel('T²',fontsize = 14)
    # # 画图Q
    # #plt.figure(figsize=(7,4))
    # scipy.io.savemat("./data_mat/SAE/TE_SAE_data.mat", {"data": Q})
    # scipy.io.savemat("./data_mat/SAE/TE_SAE_label.mat", {"limit": th_Q})
    # plt.subplot(212)
    # plt.plot(Q,'b',lw=1.5)#参数控制颜色和字体的粗细,label='mornitoring index'
    # plt.plot(th_Q,'r',lw=1) #,label='control limit'
    # #plt.plot(d2, 'ro')#每个样本点加红点
    # #plt.plot(d2.cumsum(),'r',lw=1.5)
    # plt.grid(True)
    # #plt.axis('tight')
    # plt.xlim(0,960)
    # #plt.plot(d2.cumsum(),'ro')
    # # plt.legend(loc = 0,fontsize = 14)
    # plt.title('Mornitoring performance of SAE',fontsize = 20)
    # plt.xlabel('Sample number',fontsize = 14)
    # plt.ylabel('SPE',fontsize = 14)
    # plt.show()
    #
    #
