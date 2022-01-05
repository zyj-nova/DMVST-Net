import torch

imaage_size = 20
T = 48
S = 9
window_size = 8

cnn_hidden_dim_first = 32
local_image_size = 9
batch_size = 1024
num_feature = 100
seq_len = 8
hidden_dim = 512
kernel_size = 3
feature_len = 28
conv_len = 3
toponet_len = 128
spatial_out_dim = 512
topo_embedded_dim = 64

start_year = 15
end_year = 17

len_test = 7 * 4 * 48
lr = 0.0001
nb_epoch = 100
image_size = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = 'save_models0103/model.pt'
