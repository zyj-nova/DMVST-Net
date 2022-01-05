import os
import pickle
from copy import copy

from data import *
from config.config import T

def load_stdata(filepath):
    f = h5py.File(filepath, 'r')
    data = f['data'][()]
    timestamps = f['date'][()]
    f.close()
    return data, timestamps


def remove_incomplete_days(data, timestamps, T):
    days = []
    i = 0
    t = []
    while i < len(timestamps):
        # timestamps[i] 是一个字符串 比如 '2013020134' 第9个字符开始为时段序列号 01 02 ... 48
        t.append(int(timestamps[i][8:]))
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif (i + T - 1) < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            i += 1
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)
    data = data[idx]
    timestamps = timestamps[idx]
    # print(data.shape)
    return data, timestamps


def load_meteorol(timeslots, filepath):
    f = h5py.File(filepath, 'r')
    Timeslot = f['date'][()]
    Windspeed = f['WindSpeed'][()]
    Weather = f['Weather'][()]
    Temperature = f['Temperature'][()]
    f.close()

    M = dict()
    for i, slot in enumerate(Timeslot):
        M[slot] = i
    WS = []
    WR = []
    TE = []
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(Windspeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # print("WS.min",WS.min(),"WS.max",WS.max())
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min() + 1e-4)
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min() + 1e-4)

    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])
    return merge_data


# 注意timeslots的字符串是bytes类型的
def load_holiday(timeslots, filepath):
    f = open(filepath, 'r', encoding='utf-8')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8].decode() in holidays:
            H[i] = 1
    # print("holidays:",H.sum()/48,'days')
    return H[:, None]


def load_meta_feature(timestamps, holiday_feature_file, meteorol_feature_file):
    time_feature = []
    holiday_feature = []
    meteorol_feature = []
    for t in timestamps:
        holiday_feature.append(load_holiday(t, holiday_feature_file))
        time_feature.append(timestamp2vec(t))
        meteorol_feature.append(load_meteorol(t, meteorol_feature_file))
    meta_feature = np.concatenate((holiday_feature, time_feature, meteorol_feature), axis=2)
    return meta_feature


def load_data(start_year, end_year, image_size, len_test, preprocess_name='preprocessing.pkl', nb_flow=1):
    data_all = []
    timestamps_all = []
    for year in range(start_year, end_year):
        filepath = os.path.join('datasets', 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        data, timestamps = load_stdata(filepath)

        data, timestamps = remove_incomplete_days(data, timestamps, T)

        data = data[:, :nb_flow]

        data[data < 0] = 0
        data_all.append(data)
        timestamps_all.extend(timestamps)
        print("year:{}, data shape: {}".format(year, data.shape))

    data_train = np.vstack(copy(data_all))[:-len_test]

    scaler = MinMaxNormalization()
    scaler.fit(data_train)
    data_all_mmn = [scaler.transform(d) for d in data_all]

    fpkl = open(preprocess_name, 'wb')
    pickle.dump(scaler, fpkl)
    fpkl.close()

    timestamps_all = np.array(timestamps_all)
    data_all_mmn = np.vstack(copy(data_all_mmn))

    st = STMatrix(data_all_mmn, timestamps_all, T, False)
    X_train, y_train, X_test, y_test, X_train_timestamps, y_train_timestamps, X_test_timestamps, y_test_timestamps = st.create_dataset(
        len_test, image_size)

    train_meta_feature = load_meta_feature(X_train_timestamps, './datasets/TaxiBJ/BJ_Holiday.txt',
                                           './datasets/TaxiBJ/BJ_Meteorology.h5')
    test_meta_feature = load_meta_feature(X_test_timestamps, './datasets/TaxiBJ/BJ_Holiday.txt',
                                          './datasets/TaxiBJ/BJ_Meteorology.h5')

    file = h5py.File("./TaxiBj.h5", "w")
    file.create_dataset("X_train", data=X_train)
    file.create_dataset("y_train", data=y_train)
    file.create_dataset("X_test", data=X_test)
    file.create_dataset("y_test", data=y_test)

    file.create_dataset("X_train_timestamps", data=X_train_timestamps)
    file.create_dataset("y_train_timestamps", data=y_train_timestamps)
    file.create_dataset("X_test_timestamps", data=X_test_timestamps)
    file.create_dataset("y_test_timestamps", data=y_test_timestamps)

    file.close()

    meta_file = h5py.File("./TaxiBj_Meta_Feature.h5", "w")
    meta_file.create_dataset("train_meta", data=train_meta_feature)
    meta_file.create_dataset("test_meta", data=test_meta_feature)
    meta_file.close()

    return X_train, y_train, X_test, y_test, train_meta_feature, test_meta_feature, scaler


def loadFromh5():
    f = h5py.File('./TaxiBj.h5', 'r')
    meta = h5py.File('./TaxiBj_Meta_Feature.h5', 'r')
    X_train = f["X_train"][()]
    y_train = f["y_train"][()]
    X_test = f["X_test"][()]
    y_test = f["y_test"][()]
    train_meta = meta["train_meta"]
    test_meta = meta["test_meta"]
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))
    return X_train, y_train, X_test, y_test, train_meta, test_meta, mmn

if __name__ == "__main__":
    load_data(15, 17, 20, 4 * 7 * 48)

