import logging
from config.config import *
from model.model import DMVSTNet
from lib.data import *
from lib.utils import *
from lib.load_data import loadFromh5,load_data
import h5py
import torch.nn as nn
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = 'log.txt'
lf = logging.FileHandler(log_file, mode='w')
lf.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
lf.setFormatter(formatter)
logger.addHandler(lf)

if __name__ == "__main__":
    path = os.getcwd()
    files = os.listdir(path)
    if 'TaxiBj.h5' in files:
        print("yes!")
        X_train, y_train, X_test, y_test, train_meta, test_meta, mmn = loadFromh5()
    else:
        X_train, y_train, X_test, y_test, train_meta, test_meta, mmn = load_data(
            start_year, end_year, image_size, len_test, preprocess_name='preprocessing.pkl', nb_flow=1)

    train_image_input = X_train
    train_lstm_input = np.repeat(train_meta, image_size * image_size, axis=0)

    test_image_input = X_test
    test_lstm_input = np.repeat(test_meta, image_size * image_size, axis=0)

    topo_input = h5py.File('./node_embedding.h5', 'r')['data'][()]
    train_topo_input = np.repeat(topo_input, train_meta.shape[0], axis=0)
    test_topo_input = np.repeat(topo_input, test_meta.shape[0], axis=0)

    train_data = STDataSets([train_image_input, train_lstm_input, train_topo_input], y_train)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_data = STDataSets([test_image_input, test_lstm_input, test_topo_input], y_test)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

    loss = nn.MSELoss()
    model = DMVSTNet(conv_len, spatial_out_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    min_test_loss = np.inf

    for epoch in range(nb_epoch):
        l_sum, n = 0.0, 0
        model.train()

        for data in train_iter:
            # print(x[0].shape,x[1].shape,x[2].shape,x[3].shape,y.shape)
            # break

            x = [data[0].to(device), data[1].to(device), data[2].to(device)]
            y = data[3].to(device)
            y_pred = model(x).view(len(y), -1)
            y = y.view(len(y), -1)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        test_loss = evaluate_model(model, loss, test_iter, device)
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            MAE, MAPE, RMSE = evalute_metric(model, test_iter, mmn, device)
            print("MAE: {}, MAPE: {}, RMSE: {}".format(MAE, MAPE, RMSE))
            logger.info("MAE: " + str(MAE) + ", MAPE: " + str(MAPE) + ", RMSE: " + str(RMSE))
        print("epoch:" + str(epoch) + '\t' + "train_loss:" + str(l_sum / n))
        logger.info("epoch:" + str(epoch) + '\t' + "train_loss:" + str(l_sum / n))