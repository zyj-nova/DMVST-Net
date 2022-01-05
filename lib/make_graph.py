import pickle
import numpy as np
from config.config import imaage_size

if __name__ == "__main__":
    W = pickle.load(open('./graph_weight_mmn.pkl', 'rb'))
    # because we only use the 20 × 20 nodes
    A = np.zeros((imaage_size, imaage_size))
    for i in range(A.shape[0] - 1):
        for j in range(0, len(W[i])):
            A[i, i + 1 + j] = W[i][j]
    with open('data/network_file.txt', 'w') as f:
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[1]):
                f.writelines(str(i) + ' ' + str(j) + ' ' + str(A[i, j]) + '\n')
                f.writelines(str(j) + ' ' + str(i) + ' ' + str(A[i, j]) + '\n')