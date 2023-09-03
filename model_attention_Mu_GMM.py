import torch
import torch.nn.functional as F
import torch.nn as nn



class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim1, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim1, hid_dim1)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim1, hid_dim1)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        v = torch.zeros((bsz, V.shape[2]))
        for i in range(Q.shape[1]):
            Q_i = F.softmax(Q[:, i, :], dim=-1)
            H_Q = torch.unsqueeze(torch.mean(Q_i * torch.log(Q_i), dim=1), dim=1)  # 改为pi*log（pi）计算，其中pi=softmax（[]）

            for j in range(K.shape[1]):
                K_j = F.softmax(K[:, j, :], dim=-1)
                H_K = torch.unsqueeze(torch.mean(K_j * torch.log(K_j), dim=1), dim=1)

                Q_kj = torch.cat((Q[:, i, :], K[:, j, :]), dim=1)
                Q_kj = F.softmax(Q_kj, dim=-1)

                H_Q_K = torch.unsqueeze(torch.mean(Q_kj * torch.log(Q_kj), dim=1), dim=1)
                I_Q_K1 = -(H_Q + H_K - H_Q_K) / self.scale

                if j == 0:
                    I_Q_K = I_Q_K1
                else:
                    I_Q_K = torch.cat((I_Q_K, I_Q_K1), dim=1)

            I_Q_K = torch.softmax(I_Q_K, dim=-1)  # 按列维度 3*10

            for m in range(V.shape[1]):
                v += I_Q_K[:, m].reshape(-1, 1) * V[:, m, :]  # 3 * 20
            v1 = torch.unsqueeze(v, dim=1)

            if i == 0:
                V_new = v1
            else:
                V_new = torch.cat([V_new, v1], dim=1)

        x = self.fc(V_new)
        return x


# 模型最好都在__init__中定义好，在forward中直接调用
class Multi_SAE(nn.Module):
    def __init__(self, input_size1, input_size2, input_size3):
        super(Multi_SAE, self).__init__()
        # input = []
        self.input1 = input_size1
        self.input2 = input_size2
        self.input3 = input_size3
        self.att = 16  # 隐变量统一维度，用于multihead
        self.att_out = self.att * 2  # 合并attention输出 需要*2
        # self.att_out = self.att            # 不合并att 不需要*2  与下面拼接出同改

        self.encoder1_1 = nn.Sequential(
            nn.Linear(self.input1, 30),
            nn.Tanh())
        self.encoder1_2 = nn.Sequential(
            nn.Linear(30, 24),
            nn.Tanh())
        self.att_1 = nn.Sequential(
            nn.Linear(24, self.att)  # 正常为（33，self.att）
        )

        self.encoder2_1 = nn.Sequential(
            nn.Linear(self.input2, 15),
            nn.Tanh())
        self.encoder2_2 = nn.Sequential(
            nn.Linear(15, 10),
            nn.Tanh())
        self.att_2 = nn.Sequential(  #####
            nn.Linear(10, self.att)  # 正常为(4, self.att)
        )

        self.encoder3_1 = nn.Sequential(
            nn.Linear(self.input3, 28),
            nn.Tanh())
        self.encoder3_2 = nn.Sequential(
            nn.Linear(28, 20),
            nn.Tanh())
        self.att_3 = nn.Sequential(  ######
            nn.Linear(20, self.att)  # 正常为(6, self.att)
        )
        self.all_cat_B1 = nn.Sequential(
            nn.Linear(70, 20)
        )
        self.all_cat_B2 = nn.Sequential(
            nn.Linear(41, 20)
        )
        self.all_cat_B3 = nn.Sequential(
            nn.Linear(64, 20)
        )

        # 倒数第二层插入全连接层

        self.decoder1_1 = nn.Sequential(
            nn.Linear(self.att_out, 24),
            nn.Tanh(),
            # nn.Linear(16, 30),  #15*2
            # nn.Tanh(),
            nn.Linear(24, 30),
            nn.Tanh(),
            nn.Linear(30, self.input1),

        )
        self.decoder2_1 = nn.Sequential(
            nn.Linear(self.att_out, 10),
            nn.Tanh(),
            nn.Linear(10, 15),
            # nn.Tanh(),
            # nn.Linear(10, 15),
            nn.Tanh(),
            nn.Linear(15, self.input2),

        )
        self.decoder3_1 = nn.Sequential(
            nn.Linear(self.att_out, 20),
            nn.Tanh(),
            nn.Linear(20, 28),
            nn.Tanh(),
            # nn.Linear(17, 25),
            # nn.Tanh(),
            nn.Linear(28, self.input3),
            # nn.Tanh(),
        )
        self.attention = MultiheadAttention(hid_dim1=20, hid_dim=16, n_heads=4, dropout=0.1)  ##新版本中18的时候num_heads=3,6。

    def forward(self, X1, X2, X3):
        encode1_1 = self.encoder1_1(X1)
        encode1_2 = self.encoder1_2(encode1_1)
        encode2_1 = self.encoder2_1(X2)
        encode2_2 = self.encoder2_2(encode2_1)
        encode3_1 = self.encoder3_1(X3)
        encode3_2 = self.encoder3_2(encode3_1)

        h1 = self.att_1(encode1_2)  # 维度转化为统一值
        h2 = self.att_2(encode2_2)
        h3 = self.att_3(encode3_2)

        # 各隐层拼接，统一维度

        enc_1 = torch.cat((encode1_1, encode1_2, h1), dim=1)
        enc_2 = torch.cat((encode2_1, encode2_2, h2), dim=1)
        enc_3 = torch.cat((encode3_1, encode3_2, h3), dim=1)
        fusion_1 = self.all_cat_B1(enc_1).unsqueeze(0)
        fusion_2 = self.all_cat_B2(enc_2).unsqueeze(0)
        fusion_3 = self.all_cat_B3(enc_3).unsqueeze(0)

        # attention

        h1 = h1.unsqueeze(0)  # 维度转化为统一值
        h2 = h2.unsqueeze(0)
        h3 = h3.unsqueeze(0)

        Q = torch.cat((fusion_1, fusion_2, fusion_3), axis=0)
        V = torch.cat((h1, h2, h3), axis=0)

        attn_output = self.attention(Q, Q, V)


        # out_att = attn_output
        out_att = torch.cat((attn_output, V), dim=2)  # 将引入注意力机制产生的结果与原本的结果拼接，导致输出为2倍。


        out_att_1 = out_att[0, :, :].squeeze(0)
        out_att_2 = out_att[1, :, :].squeeze(0)
        out_att_3 = out_att[2, :, :].squeeze(0)

        # 共同使用
        x_recon1 = self.decoder1_1(out_att_1)
        x_recon2 = self.decoder2_1(out_att_2)
        x_recon3 = self.decoder3_1(out_att_3)

        x_recon = torch.hstack((x_recon1, x_recon2, x_recon3))

        return x_recon, encode1_2, encode2_2, encode3_2, x_recon1, x_recon2, x_recon3  # ,encode1_1,encode2_1,encode3_1,x_re_linear1,x_re_linear2,x_re_linear3

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

