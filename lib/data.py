from utils import *
from torch.utils.data import Dataset
import torch
import numpy as np

class MinMaxNormalization(object):
    def __init__(self):
        pass

    def fit(self, x):
        self._min = x.min()
        self._max = x.max()

    def transform(self, x):
        x = 1. * (x - self._min) / (self._max - self._min)
        x = x * 2. - 1
        return x

    def fit_transform(self, x):
        self.fit(x)
        self.transform(x)
        return x

    def inverse_transform(self, y):
        y = (y + 1.) / 2
        y = 1. * y * (self._max - self._min) + self._min
        return y


class STMatrix(object):
    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        # assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, self.T)
        if CheckComplete:
            self.check_complete()
        self.make_index()

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def make_index(self):
        # mapping timestamp:index
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def get_matrix(self, timestamps):
        return self.data[self.get_index[timestamps]]

    def create_dataset(self, len_test, image_size, S = 9, window_size=8):

        # pd_timestamps -> pandas.Timestamps
        # timestamps -> str
        padding_data_all = padding_data(self.data, size=image_size, S=S)
        print("padding_data shape:{}".format(padding_data_all.shape))

        train_data = padding_data_all[:-len_test]
        test_data = padding_data_all[-len_test:]

        train_data_pd_timestamps = np.array(self.pd_timestamps[:-len_test])
        test_data_pd_timestamps = np.array(self.pd_timestamps[-len_test:])

        train_data_str_timestamps = np.array(self.timestamps[:-len_test])
        test_data_str_timestamps = np.array(self.timestamps[-len_test:])

        # silding windows
        X_train_timestamps = train_data_str_timestamps[
            np.arange(window_size)[None, :] + np.arange(train_data_str_timestamps.shape[0] - window_size)[:, None]]
        y_train_pd_timestamps = train_data_pd_timestamps[window_size:]

        X_test_timestamps = test_data_str_timestamps[
            np.arange(window_size)[None, :] + np.arange(test_data_str_timestamps.shape[0] - window_size)[:, None]]
        y_test_pd_timestamps = test_data_pd_timestamps[window_size:]

        y_train = np.array([self.get_matrix(j).squeeze(0)[:image_size, :image_size] for j in y_train_pd_timestamps]).astype('float32')
        y_test = np.array([self.get_matrix(j).squeeze(0)[:image_size, :image_size] for j in y_test_pd_timestamps]).astype('float32')
        # [len, 1, 20 * 20]
        print(y_train.shape)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        print(y_train.shape)
        X_train = train_data[
            np.arange(window_size)[None, :] + np.arange(train_data.shape[0] - window_size)[:, None]].astype('float32')

        X_test = test_data[
            np.arange(window_size)[None, :] + np.arange(test_data.shape[0] - window_size)[:, None]].astype('float32')

        # X_train [len, seq, 20 , 20, nb_flow, S, S]
        X_train = np.transpose(X_train, (0, 2, 1, 3, 4, 5))
        X_test = np.transpose(X_test, (0, 2, 1, 3, 4, 5))
        print(X_train.shape)
        X_train = X_train.reshape(-1, window_size, X_train.shape[3], X_train.shape[4], X_train.shape[5])
        X_test = X_test.reshape(-1, window_size, X_test.shape[3], X_test.shape[4], X_test.shape[5])

        return X_train, y_train, X_test, y_test, X_train_timestamps, np.array(self.timestamps[:-len_test])[window_size:], X_test_timestamps, np.array(self.timestamps[-len_test:][window_size:])


class STDataSets(Dataset):
    def __init__(self, X, y):
        self.image_input = torch.from_numpy(np.array(X[0], dtype=np.float32))
        self.lstm_input = torch.from_numpy(np.array(X[1], dtype=np.float32))
        self.topo_input = torch.from_numpy(np.array(X[2], dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.float32))

    def __getitem__(self, idx):
        return [self.image_input[idx], self.lstm_input[idx], self.topo_input[idx], self.y[idx]]

    def __len__(self):
        return len(self.y)