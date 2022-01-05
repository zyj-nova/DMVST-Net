from dtw import dtw
import pickle
import h5py

if __name__ == "__main__":
    f = h5py.File('weekly_pattern_mmn.h5','r')
    data = f['data'][()]
    W = {}
    dist = lambda x, y : abs(x - y)
    for i in range(data.shape[0]):
        W[i] = []
        for j in range(i + 1, data.shape[0]):
            w, _ , _ , _ = dtw(data[i, :], data[j, :], dist=dist)
            W[i].append(w)
    f = open('graph_weight_mmn.pkl','wb')
    pickle.dump(W,f)
    f.close()