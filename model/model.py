import torch.nn as nn

from config.config import *
from lib.load_data import *


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides = 1, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        return self.net(x)

class Local_Seq_Conv(nn.Module):
    def __init__(self, input_channel, seq_len, kernel_size):
        super(Local_Seq_Conv, self).__init__()
        self.seq_len = seq_len
        
        self.kernel_size = kernel_size
        
        self.strides = 1
        
        self.conv2d = Conv2dSame(in_channels=input_channel, out_channels=cnn_hidden_dim_first,
                                kernel_size=kernel_size, strides=1)

    def forward(self, x):
        output = []
        for i in range(self.seq_len):
            tmp = self.conv2d(x[:,i,:,:,:])
            tmp = torch.relu(tmp)
            
            output.append(tmp)
        output = torch.stack(output, dim = 1)
        return output
# x = torch.randn((batch_size, seq_len, 1,10,10))
# conv = Local_Seq_Conv(cnn_hidden_dim_first, seq_len, 5)
# conv(x).shape

class DMVSTNet(nn.Module):
    def __init__(self, conv_len, spatial_out_dim):
        super().__init__()
        # 定义局部卷积模块
        self.spatial_out_dim = spatial_out_dim
        convs = []
        for i in range(conv_len):
            if i == 0:
                convs.append(Local_Seq_Conv(1, seq_len, kernel_size).to(device))
            else:
                convs.append(Local_Seq_Conv(cnn_hidden_dim_first, seq_len, kernel_size).to(device))
        self.local_cnn = nn.Sequential(*convs)
        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(input_size = self.spatial_out_dim + feature_len, hidden_size = hidden_dim)

        self.topo_embedding = nn.Linear(in_features=toponet_len, out_features=topo_embedded_dim)
        
        self.output = nn.Linear(in_features=hidden_dim + topo_embedded_dim, out_features=1)
        
    def forward(self, x):
        image_input = x[0]
        lstm_input = torch.transpose(x[1],0,1)
        topo_input = x[2]
        
        t = self.local_cnn(image_input)

        t = self.flatten(t)  # [batch_size, -1]
        t = t.reshape(t.shape[0], seq_len, -1)
        dense = nn.Linear(in_features=t.shape[-1], out_features=self.spatial_out_dim).to(device)
        spatial_out = dense(t) # [batch_size, seq, spatial_out_dim]
        spatial_out = torch.transpose(spatial_out, 0, 1)
        
        ret = torch.cat([spatial_out, lstm_input], dim = -1)
        _, (hid, cell) = self.lstm(ret)
        lstm_out = hid.squeeze(0) # [batch_size, hidden_dim]
        
        topo_out = self.topo_embedding(topo_input)
        topo_out = torch.tanh(topo_out) # [batch_size, topo_out_dim]
        
        static_dynamic_concate = torch.cat([lstm_out, topo_out], dim = -1)
        
        out = torch.tanh(self.output(static_dynamic_concate))
        
        return out

