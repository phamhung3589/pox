import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle


# data = pd.read_csv('training_udp.csv')
# data_train = data.iloc[:, :-1]
# data_label = data.iloc[:, -1]
#
# max_train = np.max(data_train, axis=0)
# min_train = np.min(data_train, axis=0)
#
# with open('max_feature.pickle', 'wb') as handle:
#     pickle.dump(max_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('min_pickle.pickle', 'wb') as handle:
#     pickle.dump(min_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# data_train = (data_train - min_train) / (max_train - min_train)
# dataset = data_train.copy()
# dataset['AtkOrNot'] = data_label.values
#
# with open('udp_dataset.csv', 'w') as f:
#     dataset.to_csv(f, index=False)
# print data_train


class KNN_algorithm:
    def __init__(self, k, filename):
        self.k = k
        self.udp = pd.read_csv(filename)
        # self.udp = shuffle(self.udp)
        self.udp_train = self.udp.iloc[:, :-1]
        self.udp_label = self.udp.iloc[:, -1]

    def calculate(self, input_test):
        distance = np.sqrt(np.sum(np.square(self.udp_train - input_test), axis=1))
        distance = distance.sort_values()
        # print distance
        k_nearest_neighbor = distance.index[0:self.k]
        labels = self.udp_label.loc[k_nearest_neighbor]
        vote = np.count_nonzero(labels)
        if vote > self.k/2:
            label = 1
        else:
            label = 0

        return label

    def calculate_batch(self, input_batch):
        labels = [self.calculate(input_batch[i]) for i in range(len(input_batch))]

        return labels

    def accuracy(self, true_label, predict_label):
        acc = 0
        true_label = np.reshape(true_label, (1, -1))
        predict_label = np.reshape(predict_label, (1, -1))
        acc = np.count_nonzero(true_label + predict_label == 2) / len(true_label)
        return acc

    def precision(self, true_label, predict_label):
        true_label = np.reshape(true_label, (1, -1))
        predict_label = np.reshape(predict_label, (1, -1))
        TP = np.count_nonzero(true_label + predict_label == 2)
        prec = TP/ (np.count_nonzero(predict_label))
        return prec

    def recall(self, true_label, predict_label):
        true_label = np.reshape(true_label, (1, -1))
        predict_label = np.reshape(predict_label, (1, -1))
        TP = np.count_nonzero(true_label + predict_label == 2)
        rec = TP / (np.count_nonzero(true_label))
        return rec


knn = KNN_algorithm(1, './outputCSV/tcp_data_1.csv')
data = pd.DataFrame([pd.read_csv("test.csv").iloc[0,:].as_matrix()], columns=['IP_source', 'destination_port'])
with open("test.csv", "w") as f:
    data.to_csv(f, index=False)