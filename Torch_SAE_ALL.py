#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:47:55 2022

@author: yujianbo
"""
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import scipy.io as scio 
#import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import datetime
import os
for k in range(1,11):
    def load_te_detection_data(test = 5):

        path_test = 'F:\合作论文\MuSAE\data/te_data.mat'
        path_train = 'F:\合作论文\MuSAE\data/te_training_data.mat'
        data0 = scio.loadmat(path_test)# data为字典类型
        data1 = scio.loadmat(path_train)# data为字典类型
        data_training0 = data0['d00_te']# 访问字典中的一部分
        data_training1 = data1['d00']
        print(data_training0.shape)

        data_training=np.vstack((data_training0,data_training1))  #正常数据合并
        data_training_all = np.hstack((data_training[:,0:22],data_training[:,41:52]))# 提取33个变量
        # z-score 标准化
        scaler = preprocessing.StandardScaler().fit(data_training_all)# 创建标准化转换器
        X_training = preprocessing.scale(data_training_all)# 标准化处理
        #X_training = np.unsqueeze(X_training, axis=2)

        # 载入测试数据
        test = str(test)
        if len(test) == 1:
            test_str = str('0'+test)
        else:
            test_str = test
        name = str('d' + test_str + '_te')
        data_test = data0[name]
        data_test1 = np.hstack((data_test[:,0:22],data_test[:,41:52]))# 提取33个变量
        data_testing = scaler.transform(data_test1)# 测试数据标准化
        X_testing = data_testing
        #X_testing = np.unsqueeze(X_testing, axis=2)

        return X_training, X_testing,test_str


    def creat_torch_datasets(batch_size):

        valid_size = 0.2

    # =============================================================================
    #     transform = transforms.Compose([
    #
    #         transforms.ToTensor()])
    # =============================================================================

        train_data, test_data,test_str = load_te_detection_data()

        train_data = torch.FloatTensor(train_data)
        test_data = torch.FloatTensor(test_data)

        train_data = TensorDataset(train_data, train_data)
        test_data = TensorDataset(test_data, test_data)
        #train_data, test_data = np.expand_dims(train_data, axis=2), np.expand_dims(test_data, axis=2)
    # =============================================================================
    #     train_data, test_data = load_te_detection_data()
    #     train_data = torch.from_numpy(train_data).float()
    #     test_data = torch.from_numpy(test_data).float()
    # =============================================================================
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(num_train * valid_size))

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   #shuffle = True,
                                                   num_workers=0)

        valid_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   sampler=valid_sampler,
                                                   #shuffle = True,
                                                   num_workers=0)

        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  num_workers=0)

        return train_loader, valid_loader, test_loader ,test_str


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #self.fc1 = nn.Linear(33, 50)
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

            # self.encoder = nn.Sequential(  # 对应结构50-33-18-33-50,9-4-18-4-9,15-6-18-6-15
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

    autoencoder = Net()
    print(autoencoder)
    print('@' * 70)
    # summary(autoencoder, (1, 33))  # p1:model,p2:input_size

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 0.001)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=20)
    #lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    creterion = nn.MSELoss()


    def train_model(k,model, n_epochs, batch_size, patience):

        train_losses = []

        valid_losses = []

        avg_train_loss = []

        avg_valid_loss = []

        lr_his = []

        early_stopping_path = '{}-checkpoint.pt'.format(k)
        early_stopping = EarlyStopping(patience, verbose=True, path=early_stopping_path)

        #train_loader, valid_loader, test_loader = creat_torch_datasets(10)

        log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d %H%M%S'))
        # writer = SummaryWriter(log_dir = log_dir)

        #init_data = torch.zeros(1, 1 ,33)
        #writer.add_graph(model = autoencoder, input_to_model=init_data)

        for epoch in range(1, n_epochs+1):

            model.train()
            for batch, (data, _) in enumerate(train_loader):

                # data = data.cuda()

                optimizer.zero_grad()

                feature, output = model(data)

                loss = creterion(output, data)

                loss.backward()

                optimizer.step()

    # =============================================================================
    #             lr_schedule.step(loss)
    #
    #             lr = optimizer.param_groups[0]['lr']
    #
    #             lr_his.append(lr)
    # =============================================================================

                train_losses.append(loss.item())
            # writer.add_scalar(('loss/train'), loss, epoch)
            # writer.add_scalar(('lr'), optimizer.param_groups[0]['lr'], epoch)
            # writer.add_histogram('weight', autoencoder.encoder[0].weight)

            #writer.add_graph(model = autoencoder, input_to_model=data)

            model.eval()
            for batch, (data, _) in enumerate(valid_loader):

                feature, output = model(data)

                loss = creterion(output, data)

    # =============================================================================
    #             lr_schedule.step(loss)
    #
    #             lr = optimizer.param_groups[0]['lr']
    #
    #             lr_his.append(lr)
    # =============================================================================

                valid_losses.append(loss.item())
            # writer.add_scalar(('loss/valid'), loss, epoch)

            lr_schedule.step(loss)

            lr = optimizer.param_groups[0]['lr']

            lr_his.append(lr)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_loss.append(train_loss)
            avg_valid_loss.append(valid_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]' +
                         f'train_loss:{train_loss:.5f}' +
                         f'valid_loss:{valid_loss:.5f}')

            print(print_msg)

            train_losses = []
            valid_losses = []

            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print('early_stop')
                break

        # writer.close()

        model.load_state_dict(torch.load(early_stopping_path))

        return model, avg_train_loss, avg_valid_loss, lr_his


    batch_size = 10
    n_epochs = 2000

    start = time()
    train_loader, valid_loader, test_loader,test_str = creat_torch_datasets(batch_size)



    patience = 40

    model, train_loss, valid_loss, lr_his = train_model(k,autoencoder, n_epochs, batch_size, patience)


    end = time()
    run_time = end - start
    print("train_time is {:.2f}".format(run_time))
    # visualizing the loss and the early stopping checkpoint
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8),dpi=150)
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

    # find positing of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)
    plt.xlim(0, len(train_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('{}_loss_ploy.png'.format(test_str), bbox_inches='tight')


    plt.plot(lr_his)





















