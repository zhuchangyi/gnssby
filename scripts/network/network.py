import torch
import torch.nn as nn
#from torchsummary import summary
import torch.nn.functional as F
import math

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        #return self.w_2(self.dropout(nn.ReLU()(self.w_1(x))))
        return self.w_2(self.dropout(self.leaky_relu(self.w_1(x))))



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.2):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.layernorm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, attn_mask=mask)[0])
        x2 = self.layernorm2(x)
        x = x + self.dropout2(self.ffn(x2))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class GNSSClassifier(nn.Module):
    def __init__(self, d_model=64, nhead=4, d_ff=64, num_layers=3, num_satellites=81, num_classes=12, dropout=0.2):
        super(GNSSClassifier, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )

        self.transformer_encoder = TransformerEncoder(d_model, nhead, d_ff, num_layers)
        self.classifier_x = nn.Sequential(
            nn.Linear(d_model * num_satellites, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.classifier_y = nn.Sequential(
            nn.Linear(d_model * num_satellites, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        self.classifier_z = nn.Sequential(
            nn.Linear(d_model * num_satellites, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x.transpose(0, 1)  # (num_satellites, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1).contiguous().view(batch_size, -1)  # (batch_size, num_satellites * d_model)
        x_out = self.classifier_x(x)
        y_out = self.classifier_y(x)
        z_out = self.classifier_z(x)
        return x_out, y_out, z_out
class GNSSRegressor(nn.Module):
    def __init__(self, d_model=64, nhead=4, d_ff=64, num_layers=2, num_satellites=81):
        super(GNSSRegressor, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )  # 初始投影层
        self.transformer_encoder = TransformerEncoder(d_model, nhead, d_ff, num_layers)  # Transformer编码器
        self.regressor = nn.Sequential(
            nn.Linear(d_model * num_satellites, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )  # 回归器层，输出三个值(x, y, z)
        self.output_scale = nn.Parameter(torch.tensor(30.0))
        self.activation = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)  # 投影输入特征到指定维度
        x = x.transpose(0, 1)  # 调整维度以符合Transformer输入需求
        x = self.transformer_encoder(x)  # 通过Transformer编码器
        x = x.transpose(0, 1).contiguous().view(batch_size, -1)  # 调整形状以适应全连接层
        x = self.regressor(x)  # 通过回归器层获得输出
        x = self.activation(x) * self.output_scale
        return x  # 返回的output是一个[batch_size, 3]的张量，表示每个样本的(x, y, z)预测值

# # 试一下
# model = GNSSClassifier(d_model=32, nhead=4, d_ff=64, num_layers=1, num_satellites=81, num_classes=22)
# input_features = torch.randn(1, 81, 15)  # (batch_size, num_satellites, feature_dim)
# x_out, y_out, z_out = model(input_features)
# summary(model, input_size=(81, 15))
# print(x_out.shape, y_out.shape, z_out.shape)  # torch.Size([4, 21]), torch.Size([4, 21]), torch.Size([4, 21])

#




