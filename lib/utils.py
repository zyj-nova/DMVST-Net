import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def padding_data(data, size, S):
    padding_ret = []
    for i in range(len(data)):
        time_slot = []
        tmp = data[i, :, :size, :size]
        pad_size = S // 2
        padding = np.zeros((tmp.shape[0], tmp.shape[1] + 2 * pad_size, tmp.shape[2] + 2 * pad_size))
        padding[:, pad_size : pad_size + tmp.shape[1] , pad_size : pad_size + tmp.shape[2]] = tmp
        #padding之后的图， 对于每个位置生成一个 S × S的图片
        for i in range(pad_size, pad_size + tmp.shape[1]):
            for j in range(pad_size, pad_size + tmp.shape[2]):
                time_slot.append(padding[:, i - pad_size: i + pad_size + 1, j - pad_size: j + pad_size + 1])
        padding_ret.append(np.array(time_slot))
    padding_ret = np.array(padding_ret)
    # shape: [len_data, size, size, nb_flow, S, S]
    return padding_ret

def string2timestamp(strings,T=48):
    timestamps = []
    hour_per_slot = 24.0/T
    num_per_T = T // 24
    for t in strings:
        year,month,day,slot = int(t[:4]),int(t[4:6]),int(t[6:8]),int(t[8:10])-1
        timestamps.append(pd.Timestamp(datetime(
            year,month,day,hour =int(hour_per_slot*slot),minute=(slot%num_per_T)*int(60.0*hour_per_slot)
        )))
    return timestamps

def timestamp2vec(timestamps):
    vec = [time.strptime(str(t[:8],encoding='utf-8'),'%Y%m%d').tm_wday for t in timestamps]
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i>=5:
            v.append(0)
        else:
            v.append(1)
        ret.append(v)
    return np.asarray(ret)

def evaluate_model(model,loss,data_iter, device):
    model.eval()
    l_sum,n = 0.0, 0
    with torch.no_grad():
        for data in data_iter:
            x = [data[0].to(device), data[1].to(device), data[2].to(device)]
            y = data[3].to(device)
            y_pred = model(x).view(len(y),-1)
            y = y.view(len(y),-1)
            l = loss(y_pred,y)
            l_sum = l.item()*y.shape[0]
            n += y.shape[0]
        return l_sum / n
    
def evalute_metric(model,data_iter,scaler, device):
    #print("已修改")
    model.eval()
    with torch.no_grad():
        mae,mape,mse = [],[],[]
        for data in data_iter:
            x = [data[0].to(device), data[1].to(device), data[2].to(device)]
            y = data[3].to(device)
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(y),-1).cpu().numpy()).reshape(-1)
            assert len(y_pred)==len(y)

            d = np.abs(y - y_pred)
            mae += d.tolist()
            #mape += (d/y).tolist() 存在 y = 0的情况
            mape += [ d[i]/y[i] if y[i]!=0 else 0 for i in range(len(y))]
            mse += (d**2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE,MAPE,RMSE