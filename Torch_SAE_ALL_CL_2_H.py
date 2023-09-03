#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn import preprocessing
import scipy.io as scio
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from Mutual_loss import SAE_loss_CL
from model_attention_Mu_GMM import Multi_SAE



def get_index1(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]


def load_te_detection_data():
    path_test = './data/te_data.mat'
    path_train = './data/te_training_data.mat'
    data0 = scio.loadmat(path_test)  # data为字典类型
    data1 = scio.loadmat(path_train)  # data为字典类型
    data_training0 = data0['d00_te']  # 访问字典中的一部分
    data_training1 = data1['d00']
    data_training = np.vstack((data_training0, data_training1))  # 正常数据合并

    data_training_all = np.hstack((data_training[:, 0:22], data_training[:, 41:52]))  # 提取33个变量
    print(data_training.shape)
    # z-score 标准化
    scaler = preprocessing.StandardScaler().fit(data_training_all)  # 创建标准化转换器
    X_training = preprocessing.scale(data_training_all)  # 标准化处理
    X_training = X_training.T
    t0 = time()
    # 构造聚类器
    #k-Means
    #第一次
    # estimator = KMeans(n_clusters=3)
    # estimator.fit(X_training)  # 聚类
    # # joblib.dump(estimator,"estimator.pkl")
    # label_pred = estimator.labels_  # 获取聚类标签

    # AgglomerativeClustering
    # estimator = AgglomerativeClustering(linkage="ward", n_clusters=4)
    # estimator.fit(X_training)
    # label_pred = estimator.labels_

    # GMM
    estimator = GaussianMixture(n_components=3)
    estimator.fit(X_training)
    label_pred = estimator.predict(X_training)
    print(label_pred)
    print("ward : %.2fs" % (time() - t0))

    index1 = get_index1(label_pred, 0)  #
    index2 = get_index1(label_pred, 1)  #
    index3 = get_index1(label_pred, 2)  #

    X_1 = X_training[index1, :].T
    X_2 = X_training[index2, :].T
    X_3 = X_training[index3, :].T

    len1 = X_1.shape[1]
    len2 = X_2.shape[1]
    len3 = X_3.shape[1]
    print(len1)
    print(len2)
    print(len3)
    f = open("y_att.txt", "w")
    f.writelines(str(index1))
    f.writelines(str(index2))
    f.writelines(str(index3))
    f.close()
    X_training = np.hstack((X_1, X_2, X_3))
    # print(X_training.shape)

    return len1, len2, len3, X_training


def creat_torch_datasets(batch_size):
    valid_size = 0.2

    len1, len2, len3, train_data, test_data, test_str = load_te_detection_data()

    train_data = torch.FloatTensor(train_data)

    train_data = TensorDataset(train_data, train_data)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(num_train * valid_size))

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)  # sampler
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)

    return len1, len2, len3, train_loader, valid_loader

def train_model(model1, n_epochs, patience):
    train_losses = []

    valid_losses = []

    avg_train_loss = []

    avg_valid_loss = []

    lr_his = []

    early_stopping = EarlyStopping(patience,path='checkpoint.pt', verbose=True)


    for epoch in range(1, n_epochs + 1):

        autoencoder_1.train()

        for batch, (data, _) in enumerate(train_loader):
            optimizer_1.zero_grad()

            data1 = data[:, :len1]
            data2 = data[:, len1:len1 + len2]
            data3 = data[:, len1 + len2:]

            output, encode1_att, encode2_att, encode3_att, x_recon1, x_recon2, x_recon3 = model1(data1, data2, data3)

            loss = creterion(data, data1, data2, data3, x_recon1, x_recon2, x_recon3, output)

            loss.backward()

            optimizer_1.step()

            train_losses.append(loss.item())

        autoencoder_1.eval()
        for batch, (data, _) in enumerate(valid_loader):

            data1 = data[:, :len1]
            data2 = data[:, len1:len1 + len2]
            data3 = data[:, len1 + len2:]

            output, encode1_att, encode2_att, encode3_att, x_recon1, x_recon2, x_recon3 = model1(data1, data2,data3)

            loss = creterion(data, data1, data2, data3, x_recon1, x_recon2, x_recon3,output)

            valid_losses.append(loss.item())

        lr_schedule_1.step(loss)
        lr_1 = optimizer_1.param_groups[0]['lr']
        lr_his.append(lr_1)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        avg_train_loss.append(train_loss)

        avg_valid_loss.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]' +
                     f'train_loss:{train_loss:.5f}' +
                     f'valid_loss:{valid_loss:.5f}'
                     )

        print(print_msg)

        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model1)

        if early_stopping.early_stop:
            print('early_stop')
            break

    return model1, avg_train_loss, avg_valid_loss, lr_his


batch_size = 25  #10\16\32\64\80
n_epochs = 2000

start = time()

len1, len2, len3, train_loader, valid_loader, test_loader, test_str = creat_torch_datasets()

print("train_loader 的长度是 {}".format(len(train_loader)))
print("valid_loader 的长度是 {}".format(len(valid_loader)))

autoencoder_1 = Multi_SAE(len1, len2, len3)

print(autoencoder_1)
print('@' * 70)

optimizer_1 = torch.optim.Adam(autoencoder_1.parameters(), lr=0.001)

lr_schedule_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, factor=0.2, patience=20)

creterion = SAE_loss_CL()

patience = 20

model1, train_loss, valid_loss, lr_his = train_model(autoencoder_1, n_epochs, patience)

end = time()
run_time = end - start
print("train_time is {:.2f}".format(run_time))
# visualizing the loss and the early stopping checkpoint
# visualize the loss as the network trained
fig = plt.figure(figsize=(10, 8), dpi=150)
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

# find positing of lowest validation loss
minposs = valid_loss.index(min(valid_loss)) + 1
plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim(0, 0.5)
plt.xlim(0, len(train_loss) + 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_ploy.png', bbox_inches='tight')

plt.plot(lr_his)
