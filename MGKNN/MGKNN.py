from Naturalneighbor import *
from GranularBallGeneration import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class MGKNN():
    def __init__(self, data):
        self.data_ = data

    def getDistance(self, X):
        # An efficient method to obtain n*n matrix distance
        m, n = X.shape
        G = np.dot(X.T, X)
        H = np.tile(np.diag(G), (n, 1))
        A = H + H.T - G * 2
        return A

    def mgknn(self):

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(self.data_)
        gb_cluster = GBC(data)

        group = {}
        for i in range(len(gb_cluster)):
            label = gb_cluster[i].label
            if(label in group.keys()):
                group[label] = np.append(group[label], gb_cluster[i].data, 0)
            else:
                group[label] = gb_cluster[i].data

        gbnc = {}
        for key in group.keys():
            D1 = group[key]
            D2 = D1.T
            A = self.getDistance(D2)
            NNtool = NNSearch(A)
            t, nn, rnn, dis_index = NNtool.natural_search()
            gbnc[key] = t + 1

        adaptive_k = []
        for d in data:
            for key, value in group.items():
                    if np.any(np.all(d == value, axis=1)):
                        adaptive_k.append(hbnc[key])

        return adaptive_k
