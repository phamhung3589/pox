import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
import time as t

class LocalOutlierFactor:
    def __init__(self, k, X1):
        self.k          = k
        self.X1         = X1
        self._threshold = []
        self._data      = pd.DataFrame()
        self._k_dist    = []
        self.set_knn    = []
        self._lrd       = []

    def set_x(self, threshold):
        self._threshold = threshold

    def set_data(self, data):
        self._data = data

    def set_k_dist(self, k_dist):
        self._k_dist = k_dist

    def set_set_knn(self, set_knn):
        self._set_knn = set_knn

    def set_lrd(self, lrd):
        self._lrd = lrd

    def print_percent(self,i, len_X):
        percent = [np.floor(len_X*k) for k in np.arange(0.1,1.01,0.1)]
        for idx, val in enumerate(percent):
            if i == val:
                print("{}%".format((idx+1)*10))

    def k_distance(self):
        k_dist = []
        set_knn = []
        for i in self.X1.index:
            # Compute distance of input i to all elements in data
            X_new = np.sqrt(np.sum(np.power(self.X1.drop(index=i) - self.X1.loc[i,:], 2), axis=1))
            X_new.sort_values(inplace=True)
            k_dist.append(X_new.index[self.k-1])
            set_knn.append(X_new.index[:self.k])
        self.set_k_dist(k_dist)
        self.set_set_knn(set_knn)
        return k_dist, set_knn

    def reachability_distance(self, kth_dist, dist):
        return dist
        # return max(kth_dist, dist)

    def euclidean_distance(self, A, B):
        return np.sqrt(np.sum(np.power(A - B, 2)))

    def local_reachability_density(self):
        kth_dist, set_knn = self.k_distance()
        lrd = []
        for i in range(len(self.X1.index)):
            # density = [sum(self.reachability_distance(set_knn[i,j], self.euclidean_distance(self.X1.loc[i,:], self.X1.loc[j]))) for j in set_knn[i]]
            density = 0
            for idx in set_knn[i]:
                dist_idx = self.euclidean_distance(self.X1.iloc[i], self.X1.loc[idx])
                kth_idx = kth_dist[self.X1.index.get_loc(idx)]
                density += self.reachability_distance(kth_idx, dist_idx)
            # if density != 0:
            lrd.append(len(set_knn[i])/density)
        self.set_lrd(lrd)
        return lrd

    # After computing local reachability density of all points, these values are then compared with those
    # of the neighbors using this formula
    def LOF(self):
        # print("The process of training is begining...")
        len_X1 = len(self.X1.index)
        lrd = self.local_reachability_density()
        outliers = []
        percent = 10
        for i in range(len_X1):
            # self.print_percent(i, len_X1)
            N = len(self._set_knn[i])
            sum_density = 0
            for idx in self._set_knn[i]:
                sum_density += lrd[self.X1.index.get_loc(idx)]
            outliers.append(sum_density/(N*lrd[i]))
            self.set_x(outliers)
        # print self._threshold
        # print("The process of training has done")
        return outliers

    # Predict the new value after
    def k_distance_predict(self):
        k_dist = []
        set_knn = []
        L = len(self._data)
        for i in range(1):
            # Compute distance of input i to all elements in data
            # X_new = np.sqrt(np.sum(np.power(self.X1 - self._data.iloc[i, :], 2), axis=1))
            X_new = np.sqrt(np.sum(np.power(self.X1 - self._data, 2), axis=1))
            X_new.sort_values(inplace=True)
            k_dist.append(X_new.index[self.k-1])
            set_knn.append(X_new.index[:self.k])
        return k_dist, set_knn

    def local_reachability_density_predict(self):
        kth_dist, set_knn = self.k_distance_predict()
        # kth_dist_test, set_knn_test = self.k_distance()
        lrd = []
        L = len(self._data)
        for i in range(1):
            # density = [sum(self.reachability_distance(set_knn[i,j], self.euclidean_distance(self.X1.loc[i,:], self.X1.loc[j]))) for j in set_knn[i]]
            density = 0
            for idx in set_knn[i]:
                dist_idx = self.euclidean_distance(self._data, self.X1.loc[idx])
                kth_idx = self._k_dist[self.X1.index.get_loc(idx)]
                density += self.reachability_distance(kth_idx, dist_idx)
            lrd.append(len(set_knn[i])/density)

        return lrd

    def LOF_predict(self):
        kth_dist, set_knn = self.k_distance_predict()
        lrd = self.local_reachability_density_predict()
        outliers = []
        L = len(self._data)

        for i in range(1):
            N = len(set_knn[i])
            sum_density = 0
            for idx in set_knn[i]:
                sum_density += self._lrd[self.X1.index.get_loc(idx)]
            outliers.append(sum_density/(N*lrd[i]))
        return outliers

    def predict(self, data):
        self.set_data(data)
        max_lof = sorted(self._threshold,reverse=True)
        print("max of the data: ", max_lof[0])
        outliers = self.LOF_predict()
        print "outliers", outliers
        label = [int(i > max_lof[0]) for i in outliers]
        return label

    def score(self, y_predict, y_true):
        result = [y_predict[i] - y_true[i] for i in range(len(y_predict))]
        return  result.count(0)/len(y_true)


#### Preprocessing the data into Dataframe and take these parameter in the input of Algorithm
# col_names = ['IP_source', 'port_source', 'port_destination', 'packet_type', 'total_packet', 'Atk or Not']
data = pd.read_csv('./outputCSV/normal_udp1.csv')
# data = shuffle(data)
# data.columns = ['ent_ip_src', 'ent_tp_src', 'ent_tp_dst', 'ent_nw_proto', 'packet_count', 'atkOrNot']
# data = data[['ent_ip_src', 'packet_count']]
#
# test = pd.read_csv('mini_test_lof.csv', header=None, names=col_names)
# test = shuffle(test)
# test_data = test.iloc[:,[0,1,2,3,4]]
# test_label = test.iloc[:,-1]
#
# X = data.iloc[:,[0,4]]
# y = data.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# print(X.drop(index=[0,1]))

# The process of testing data with the kth-nearest neighbors = 11
print("The process of training is begining...")
lof1 = LocalOutlierFactor(20, data)
dist = lof1.LOF()
# print("The process of training has done")
#
# predict = lof1.predict(test_data)
# print("Score: {}".format(lof1.score(predict, test_label.as_matrix())))

##### Visualization the data
# plt.figure()
# plt.scatter(X.iloc[y.as_matrix() == 0, 0], X.iloc[y.as_matrix() == 0, 1], marker='o', c='g', linewidths=1, label="Training Normal")
# plt.scatter(test_data.iloc[test_label.as_matrix() == 0, 0], test_data.iloc[test_label.as_matrix() == 0, 1], marker='o', c='m', linewidths=1, label="Test Normal")
# plt.scatter(test_data.iloc[test_label.as_matrix() == 1, 0], test_data.iloc[test_label.as_matrix() == 1, 1], marker='o', c='r', linewidths=1, label="Test Attack")
# plt.xlabel("IP source", fontsize=15)
# plt.ylabel("Total packet", fontsize=15)
# plt.title("Classification Algorithm", fontsize=15)
# plt.legend()
# plt.show()

##### Test data with the different kth- nearest neighbors from 1 - 20
# scores = []
# for i in range(1,20,1):
#     n1 = t.time()
#     print("k neighbors = {}".format(i))
#     lof = LocalOutlierFactor(i, X_train)
#     d = lof.LOF()
#     p = lof.predict(test_data)
#     score = lof.score(p, test_label.as_matrix())
#     print("Score = {}".format(score))
#     scores.append(score)
#     n2 = t.time()
#     print("{}".format(n2 - n1))
#
# plt.figure()
# plt.plot(range(1, 20, 1), scores,'g-', linewidth=2, label="Accuracy")
# plt.xlabel("Kth Nearest Neighbors ", fontsize=18)
# plt.ylabel("Accuracy of the data",fontsize=18)
# plt.title("Anomaly detection - learning from normal data",fontsize=18)
# plt.legend()
# plt.xticks(range(1, 20, 1))
# plt.axis([0, 20, 0, 1.1])
# plt.show()

# plt.figure()
# # plt.scatter(X.iloc[:,0], X.iloc[:,1], marker='o')
# plt.plot(dist)
# plt.show()
