import math
from collections import defaultdict, Counter
from warnings import warn
import numpy as np
import sys
from algorithm import Algorithm
# knn, phong
# som knn, by vote
# som distributed , distributed center
# som one type radius (validation)


"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return math.sqrt(np.dot(x, x.T))


class MiniSom(Algorithm):
    def __init__(self, x, y, training_data, sigma=1.0, learning_rate=0.5, decay_function=None, random_seed=None, only_type = None):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            training_data - the training data, used for random initialization
            sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
            decay_function, function that reduces learning_rate and sigma at each iteration
                            default function: lambda x,current_iteration,max_iter: x/(1+current_iteration/max_iter)
            random_seed, random seed to use.
        """
        if sigma >= x/2.0 or sigma >= y/2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        if random_seed:
            self.random_generator = np.random.RandomState(random_seed)
        else:
            self.random_generator = np.random.RandomState(random_seed)
        if decay_function:
            self._decay_function = decay_function
        else:
            self._decay_function = lambda x, t, max_iter: x/(1+t/max_iter)
        self.learning_rate = learning_rate
        self.sigma = sigma
        # self.weights = self.random_generator.rand(x,y,training_data.shape[1])*1
        self.weights = np.zeros((x,y,training_data.shape[1]))
        for col in range(training_data.shape[1]):
            my_min = np.min(training_data[:,col])
            my_max = np.max(training_data[:,col])
            self.weights[:,:,col] = np.random.uniform(my_min, my_max, size=(x,y))
        self.activation_map = np.zeros((x,y))
        self.neigx = np.arange(x)
        self.neigy = np.arange(y) # used to evaluate the neighborhood function
        self.neighborhood = self.gaussian
        if only_type != None:
            self.only_type = only_type

    def _activate(self, x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        s = np.subtract(x, self.weights) # x - w
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])  # || x - w ||
            it.iternext()

    def activate(self, x):
        """ Returns the activation map to x """
        self._activate(x)
        return self.activation_map

    def gaussian(self, c, sigma):
        """ Returns a Gaussian centered in c """
        d = 2*np.pi*sigma*sigma
        ax = np.exp(-np.power(self.neigx-c[0], 2)/d)
        ay = np.exp(-np.power(self.neigy-c[1], 2)/d)
        return np.outer(ax, ay)  # the external product gives a matrix

    def diff_gaussian(self, c, sigma):
        """ Mexican hat centered in c (unused) """
        xx, yy = np.meshgrid(self.neigx, self.neigy)
        p = np.power(xx-c[0], 2) + np.power(yy-c[1], 2)
        d = 2*np.pi*sigma*sigma
        return np.exp(-p/d)*(1-2/d*p)

    def winner(self, x):
        """ Computes the coordinates of the winning neuron for the sample x """
        self._activate(x)
        return np.unravel_index(self.activation_map.argmin(), self.activation_map.shape)

    def winners(self,x):
        """
        return sorted indices with the first element being the nearest neurons
        """
        self._activate(x)
        return tuple(map(tuple,np.dstack(np.unravel_index(np.argsort(self.activation_map.ravel()), self.activation_map.shape))[0]))

    def update(self, x, win, t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            t - iteration index
        """
        eta = self._decay_function(self.learning_rate, t, self.T)
        sig = self._decay_function(self.sigma, t, self.T) # sigma and learning rate decrease with the same rule
        neighborhood = self.neighborhood(win, sig)
        g = neighborhood * eta # improves the performances

        # print('Ln thu:',t, '\tWin:', win, '\tLearn rate:',eta, '\tRadius sigma:', sig)
        # radius = 2
        # left = max(win[0]-radius, 0)
        # bottom = max(win[1]-radius, 0)
        # right = min(win[0]+radius, self.weights.shape[0]-1)
        # top = min(win[1]+radius, self.weights.shape[1]-1)
        # print('Neighborhood:')
        # print(neighborhood[left:right+1, bottom:top+1])
        # print('Rate:')
        # print(g[left:right+1, bottom:top+1])
        # print()

        it = np.nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])
            # normalization
            # self.weights[it.multi_index] = self.weights[it.multi_index] / fast_norm(self.weights[it.multi_index])
            it.iternext()

    def quantization(self, data):
        """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
        q = np.zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """ Initializes the weights of the SOM picking random samples from data """
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[self.random_generator.randint(len(data))]
            self.weights[it.multi_index] = self.weights[it.multi_index]/fast_norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self, data, num_epoch):
        """ Trains the SOM picking samples at random from data """
        num_iteration = int(len(data) * num_epoch)
        self._init_T(num_iteration)
        next_target = 0
        for iteration in range(num_iteration):
            percent = (iteration+1)/num_iteration*100
            if percent >= next_target:
                print str(int(percent))+'%'
                sys.stdout.flush()
                next_target +=5
            rand_i = self.random_generator.randint(len(data)) # pick a random sample
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)
        print()

    def train_batch(self, data, num_iteration):
        """ Trains using all the vectors in data sequentially """
        self._init_T(len(data)*num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data)-1)
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1

    def _init_T(self, num_iteration):
        """ Initializes the parameter T needed to adjust the learning rate """
        self.T = num_iteration/2  # keeps the learning rate nearly constant for the last half of the iterations

    def distance_map(self):
        """ Returns the distance map of the weights.
            Each cell is the normalised sum of the distances between a neuron and its neighbours.
        """
        um = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += fast_norm(self.weights[ii, jj, :]-self.weights[it.multi_index])
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data):
        """
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.
        """
        error = 0
        for x in data:
            error += fast_norm(x-self.weights[self.winner(x)])
        return error/len(data)

    def construct_winmap(self, data):
        """
            Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
            that have been mapped in the position i,j.
        """
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x[:-1])].append(x)
        self.winmap = winmap
        return self.winmap

    def what_type_knn(self, test_instance):
        nearests = self.winners(test_instance)
        samples = 0
        for i, index in enumerate(nearests):
            samples += len(self.winmap[index])
            if i>= self.k-1 and samples >0:
                needed_neuron = i+1
                break
        counter = Counter(sample[-1] for i in range(needed_neuron) for sample in self.winmap[nearests[i]])

        # if needed_neuron > self.k:
        #     self.cnt +=1
            # print('need:', needed_neuron)
            # print(counter)
            # print()

        return counter.most_common()[0][0]

    def set_threshold_probability(self, prob_threshold):
        temp = np.reshape(self.weights, (self.weights.shape[0] * self.weights.shape[1], self.weights.shape[2]))
        hist_values, base = np.histogram(pair_distances, bins=1000)
        cumu = np.cumsum(hist_values) / len(pair_distances)
        threshold_index = np.searchsorted(cumu, prob_threshold)
        if threshold_index == len(cumu):
            threshold_index -=1
        print 'ung voi', prob_threshold,'la range =', base[threshold_index]
        # plt.plot(base[:-1], cumu, c='blue')
        # weights = np.ones_like(distances)/len(distances)
        # plt.hist(pair_distances, bins = 100, weights=weights) # just the distribution, not cumulative
        self.threshold = base[threshold_index]

    def set_center_threshold_probability(self, prob_threshold):
        temp = np.reshape(self.weights, (self.weights.shape[0] * self.weights.shape[1], self.weights.shape[2]))
        center = temp.mean(axis = 0)
        # the distances to the center
        distances = np.apply_along_axis(np.linalg.norm, 1, temp - center)
        # distribution of the distances to the center
        hist_values, base = np.histogram(distances, bins=1000)
        cumu = np.cumsum(hist_values) / len(distances)
        # print(cumu)
        threshold_index = np.searchsorted(cumu, prob_threshold)
        if threshold_index == len(cumu):
            threshold_index -=1
        print 'ung voi', prob_threshold,'la range =', base[threshold_index]
        # mysize = (4,3.5)
        # fig1 = plt.figure(figsize=mysize)
        # plt.plot(base[:-1], cumu, c='blue', linewidth = 3)
        # plt.locator_params(axis='x',nbins=5)
        # plt.locator_params(axis='y',nbins=5)
        # fig1.savefig('cumu-distance.pdf')
        #
        # fig2 = plt.figure(figsize=mysize)
        # weights = np.ones_like(distances)/len(distances)
        # plt.hist(distances, bins = 100, weights=weights, linewidth = 3) # just the distribution, not cumulative
        # plt.locator_params(axis='x',nbins=5)
        # plt.locator_params(axis='y',nbins=5)
        # fig2.savefig('hist-distance.pdf')
        self.center = center
        self.center_threshold = base[threshold_index]

    def what_type_center_threshold(self, test_instance):
        distance = fast_norm(self.center - test_instance)
        if distance <= self.center_threshold:
            return self.only_type
        else:
            return int(not self.only_type)

    def what_type_threshold(self, test_instance):
        winner = self.winner(test_instance)
        distance = fast_norm(self.weights[winner[0]][winner[1]] - test_instance)
        if distance <= self.threshold:
            return self.only_type
        else:
            return int(not self.only_type)

    def train_distance(self, validation_data):
        dec = 100
        max_accuracy = -1
        for i in range(1,dec):
            self.threshold = i/dec
            # print(self.threshold)
            correct = 0
            for sample in validation_data:
                correct += self.what_type_threshold(sample[:-1]) == sample[-1]
            accuracy = correct / len(validation_data)
            # print(accuracy)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                # print('best', max_accuracy*100, '% with threshold', self.threshold)
                best_threshold = self.threshold
        self.threshold = best_threshold
        print('chon:', self.threshold)
        # print()

    def set_classify_method(self, method, training_data = None, k = 3, prob_threshold = 0.95, validation_data = None):
        if method == 'knn':
            self.construct_winmap(training_data)
            self.k = k
            self.what_type = self.what_type_knn
            # som.cnt = 0
        elif method == 'distribution' or method == 'validation':
            if self.only_type == None:
                raise Exception('SOM must be trained with only one type')
            self.what_type = self.what_type_threshold
            if method == 'distribution':
                self.set_threshold_probability(prob_threshold)
            else:
                self.train_distance(validation_data)
        elif method == 'distribution_center':
            if self.only_type == None:
                raise Exception('SOM must be trained with only one type')
            self.what_type = self.what_type_center_threshold
            self.set_center_threshold_probability(prob_threshold)
        else:
            self.what_type = None


import readFile
import pandas
# df = pandas.read_csv('output/normalized-1-5.csv')
# a = df.replace('normal',0)
# a = a.replace('attack',1)
# df = pandas.read_csv('output/normalized-6.csv')
# b = df.replace('normal',0)
# b = b.replace('attack',1)
# my_data = np.concatenate((a.values[14000:],b.values))
my_data = pandas.read_csv('outputCSV/normalizedFeature.csv').values

# from sklearn.datasets import load_iris
# data = load_iris()
# my_data = data.data[np.logical_or(data.target ==0, data.target ==1)]
# my_data_result = np.array([data.target[np.logical_or(data.target ==0, data.target ==1)]]).T
# my_data = np.append(my_data, my_data_result, axis = 1)

config = {}
config['method'] = 'distribution_center'
if config['method'] == 'knn':
    config['train_type'] = None
    training_data, test_data = readFile.split(my_list = my_data)
    kargs = {'k': 3, 'training_data':training_data}
elif config['method'] == 'distribution' or config['method'] == 'distribution_center':
    config['train_type'] = 1
    training_data, test_data = readFile.split(my_list = my_data,only_type = config['train_type'])
    print "TEST DATA = \n", len(test_data)
    kargs = {'prob_threshold': 0.95}
elif config['method'] == 'validation':
    config['train_type'] = 1
    training_data, test_data = readFile.split(my_list = my_data, only_type = config['train_type'])
    validation_data, test_data = readFile.split(test_data, ratio = 0.4)
    kargs = {'validation_data': validation_data}

training_weight_data = training_data[:,:-1]
training_result_data = training_data[:,-1]
normal_training = training_data[training_data[:,-1] ==0]
atack_training = training_data[training_data[:,-1] ==1]
setosa = training_weight_data[training_result_data ==0]
versi= training_weight_data[training_result_data ==1]
virgin = training_weight_data[training_result_data ==2]

som = MiniSom(30,30,training_weight_data,sigma=3,learning_rate=0.5, only_type = config['train_type'])

# import Axes3D from mpl_toolkits.mplot3d 
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(18, 10))
# markers = ['o','x','s']
# colors = ['b','r','g','y']

# def plot_3d(position, data_list, label_list, col1, col2, col3):
#     ax = fig.add_subplot(position, projection='3d', adjustable='box', aspect=1)
#     for i,x in enumerate(data_list):
#         ax.plot(x[: ,col1],x[: ,col2],x[: ,col3],c=colors[i],label=label_list[i])
#     ax.scatter(som.weights[:,:,col1].ravel(), som.weights[:,:,col2].ravel(), som.weights[:,:,col3].ravel(), c=colors[len(data_list)],label='neuron')
#     plt.legend(loc='upper left')

# def plot_2d(position):
#     ax = fig.add_subplot(position, adjustable='box', aspect=1)
#     plt.bone()
#     plt.pcolor(som.distance_map().T) # plotting the distance map as background
#     plt.colorbar()
#     t = training_result_data.astype(int)
#     # use different colors and markers for each label
#     for cnt,xx in enumerate(training_weight_data):
#         w = som.winner(xx)
#         # palce a marker on the winning position for the sample xx
#         plt.plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None', markeredgecolor=colors[t[cnt]],markersize=10,markeredgewidth=2)
#     plt.axis([0,som.weights.shape[0],0,som.weights.shape[1]])

# plot_3d(221, [normal_training, atack_training], ['normal', 'attack'], 1,2,3)
# plot_2d(111)
som.train_random(training_weight_data,5) # random training
# plot_3d(223, [normal_training, atack_training], ['normal', 'attack'], 1,2,3)
# plot_2d(111)

som.set_classify_method(config['method'], **kargs)
print "DA CHAY XONG ROI"
# filename = 'output/som'
# import os
# os.makedirs(os.path.dirname(filename), exist_ok=True)
# import pickle
# with open(filename, 'wb') as outfile:
#     pickle.dump(som, outfile)

# som.test(test_data)
# som.what_type(...)

# fig.set_tight_layout(True)
# plt.show()
# fig.savefig('ha.pdf')

#final som knn
# t_accuracy = []
# t_time = []
# t_k = []
# max_no = 300
# t_accuracy= [97.376311844077961, 97.226386806596693, 97.226386806596693, 97.301349325337327, 97.07646176911544, 97.151424287856074, 97.451274362818594, 97.376311844077961, 97.526236881559228, 97.301349325337327, 97.376311844077961, 97.601199400299848, 97.751124437781115, 97.751124437781115, 97.451274362818594, 97.526236881559228, 97.751124437781115, 97.751124437781115, 97.826086956521735, 97.751124437781115, 97.751124437781115, 97.751124437781115, 97.751124437781115, 97.751124437781115, 97.901049475262369, 97.826086956521735, 97.751124437781115, 97.826086956521735, 97.826086956521735, 97.826086956521735, 97.901049475262369, 97.901049475262369, 97.901049475262369, 97.901049475262369, 97.901049475262369, 97.901049475262369, 97.901049475262369, 97.976011994003002, 97.976011994003002, 98.050974512743622, 98.050974512743622, 97.976011994003002, 97.826086956521735, 97.901049475262369, 98.125937031484256, 98.275862068965509, 98.275862068965509, 98.275862068965509, 98.275862068965509, 98.200899550224889, 98.200899550224889, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.200899550224889, 98.200899550224889, 98.200899550224889, 98.200899550224889, 98.200899550224889, 98.200899550224889, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 97.976011994003002, 98.050974512743622, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.050974512743622, 98.125937031484256, 98.050974512743622, 98.050974512743622, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256, 98.125937031484256]
# t_time= [2.6891154685299257, 2.670310426509005, 2.753958709236385, 3.035240623725765, 2.861055477090861, 2.863252716979523, 3.0991725478393923, 3.00842729108087, 3.0307855563185204, 3.037291666914498, 2.993743637691195, 3.1263485126409574, 3.131269872456655, 3.2085180282592773, 3.1032953662672145, 3.0220359161697226, 3.0741289459068377, 3.080940854245576, 3.0696694103852917, 3.0428380265586203, 3.278166457809608, 3.2444850973103536, 3.322170234691614, 3.3825891486172197, 3.382613276434445, 3.280064512764675, 3.3573207111730388, 3.312061394172451, 3.3436498899331157, 3.31507486918162, 3.25995728351187, 3.4226574104229015, 3.1720818072065957, 3.275796212535212, 3.3064594511864245, 3.2667675833294596, 3.25048648971489, 3.2944579174493565, 3.343931738583223, 3.375491817136933, 3.427427926699797, 3.28733431345698, 3.2740532905086766, 3.4109827639280947, 3.3565345017806343, 3.2303238558447522, 3.2105000837632027, 3.2213418737523023, 3.045842565339187, 3.132409420327983, 3.151620703301151, 3.136958675584693, 3.1316896964763776, 3.127883756714782, 3.143785596787483, 3.044856899324386, 2.909590636772373, 3.2230127697763056, 3.122558479366274, 3.0977175510984134, 3.198948935947676, 3.222378654994707, 3.4595937028281516, 3.34845793837014, 3.3510104827080176, 3.383335144980915, 3.375757044401841, 3.37377802721564, 3.3641438791598155, 3.3717729162419214, 3.2834452667693865, 3.305654296989384, 3.234648454314408, 3.3301334867234353, 3.2876687071312674, 3.267095900368297, 3.2475436109116766, 3.2906619862637956, 3.3662564036013305, 3.2874558461659675, 3.3178913897124005, 3.322040480652313, 3.358824678446757, 3.2896193071105135, 3.3094017938159213, 3.2639549947392634, 3.344844127523488, 3.339578722906613, 3.3560982351002844, 3.4214061596940484, 3.464746689689213, 3.4863854097998304, 3.32142960006508, 3.369001613027867, 3.338817892403438, 3.3933957298656274, 3.4437256416995665, 3.36426076502993, 3.3289798196109164, 3.3616093860037144, 3.341285542569597, 3.3645488690400587, 3.345516489363503, 3.5109573575867703, 3.5001461294994898, 3.342502299456046, 3.5228604140846445, 3.345522565998952, 3.3837526455812013, 3.6113650008835, 3.3443644307721323, 3.4154197801535635, 3.53353009231087, 3.634808839171723, 3.6983749140863833, 3.5812945916377443, 3.5803687983545767, 3.539589212752175, 3.603762951211772, 3.622293114840895, 3.5950839072689305, 3.5427855229985408, 3.545469251172296, 3.540474614282062, 3.6151820215685615, 3.48715803612476, 3.6440172652969474, 3.5795622143609593, 3.5800120641087845, 3.6967932016238283, 3.6852540283546276, 3.6319437055573474, 3.515490170182853, 3.569289304684663, 3.57183130427279, 3.591030791424204, 3.5968793743196454, 3.574296809744084, 3.5624865112990993, 3.502804300059443, 3.5422736558242183, 3.555781837703585, 3.6134054993224822, 3.6099857833610662, 3.526449739664927, 3.6032480457197242, 3.7558643535516785, 3.6286124582590906, 3.5235434993989823, 3.629721801677744]
# # for i in range(1,max_no, 2):
# #     som.k = i
# #     print(i, end=': ')
# #     accu, time = som.test(test_data)
# #     t_accuracy.append(accu)
# #     t_time.append(time)
# #     t_k.append(i)
#
# print("t_accuracy =", t_accuracy)
# print("t_time =", t_time)
# t_k = np.arange(1, max_no, 2)
# # print(t_k.shape, t_time.shape, t_accuracy.shape)
#
# fig_size = (4.5,4.5)
# fig, ax1 = plt.subplots(figsize=fig_size)
#
# line1, = ax1.plot(t_k[:40], t_accuracy[:40], linewidth=3, label='Accuracy')
# ax1.set_xlabel('k')
# # ax1.set_ylabel('Accuracy (%)')
# ax1.set_ylim([40,100])
# for t in ax1.get_yticklabels():
#     t.set_color('b')
#
# ax2 = ax1.twinx()
# line2, = ax2.plot(t_k[:40], t_time[:40], 'r', linewidth = 3, label='Processing time')
# # ax2.set_ylabel('Processing time (ms)')
# ax2.set_ylim([2.2,4.5])
# plt.legend(handles=[line1,line2], loc='lower right')
# for t in ax2.get_yticklabels():
#     t.set_color('r')
#
# fig.savefig('ha.pdf')
#
# plt.show()

# # final distributed center
# som.set_classify_method(config['method'], **kargs)
# # som.test(test_data)
# t_accuracy = []
# t_time = []
# t_k = []
# t_threshold = []
# max_no = 100
# # for i in range(1,max_no, 1):
# #     som.set_center_threshold_probability(i/100)
# #     print(i, end=': ')
# #     accu, time = som.test(test_data)
# #     t_accuracy.append(accu)
# #     t_time.append(time)
# #     t_k.append(i)
# # print("t_accuracy =", t_accuracy)
# # print("t_time =", t_time)
# t_k = np.arange(1, max_no, 1)
# # print(t_k.shape, t_time.shape, t_accuracy.shape)
# t_accuracy = [47.826086956521742, 48.200899550224882, 48.650674662668663, 48.800599700149924, 49.175412293853071, 49.400299850074965, 49.700149925037479, 50.074962518740627, 50.14992503748126, 50.674662668665668, 51.199400299850076, 51.949025487256371, 52.173913043478258, 53.073463268365813, 53.448275862068961, 53.898050974512742, 54.047976011994002, 54.722638680659671, 54.947526236881558, 55.397301349325332, 55.69715142428786, 56.521739130434781, 56.896551724137936, 57.421289355322337, 57.646176911544224, 57.721139430284865, 57.946026986506752, 58.470764617691152, 58.920539730134934, 59.295352323838081, 59.370314842578708, 60.11994002998501, 60.269865067466263, 60.49475262368815, 60.794602698650678, 60.869565217391312, 61.019490254872565, 61.469265367316339, 61.6191904047976, 61.994002998500754, 62.368815592203894, 62.668665667166415, 63.193403298350823, 63.643178410794597, 64.017991004497759, 64.542728635682153, 65.142428785607194, 65.592203898050968, 66.266866566716644, 66.416791604197897, 66.641679160419784, 67.016491754122939, 67.541229385307346, 67.61619190404798, 68.065967016491754, 68.665667166416782, 69.11544227886057, 69.190404797601204, 69.49025487256371, 69.790104947526231, 70.164917541229386, 70.689655172413794, 71.439280359820089, 72.03898050974513, 72.938530734632678, 74.137931034482762, 74.662668665667169, 75.487256371814098, 76.236881559220393, 76.46176911544228, 76.46176911544228, 77.136431784107955, 78.110944527736137, 79.010494752623686, 79.535232383808093, 80.284857571214403, 81.634182908545725, 83.283358320839582, 83.658170914542723, 85.157421289355312, 85.607196401799101, 86.056971514242875, 86.056971514242875, 87.331334332833592, 87.931034482758619, 89.280359820089956, 89.880059970014997, 91.979010494752629, 92.503748125937037, 93.403298350824599, 94.302848575712133, 94.977511244377808, 95.502248875562231, 96.026986506746624, 96.476761619190412, 97.451274362818594, 97.451274362818594, 97.301349325337327, 97.451274362818594]
# t_time = [0.005925077012275112, 0.005052364927003052, 0.004763188569442086, 0.009144085279290287, 0.006329173269657895, 0.004820022983350854, 0.005302758052431304, 0.005378537270976328, 0.004908491646510729, 0.00488597235043367, 0.005565304448758287, 0.004584821446546014, 0.004946917429499362, 0.0058096209387371745, 0.006125248532960083, 0.005678437102859703, 0.005511329628002161, 0.0057143607418397736, 0.0068245977833531965, 0.005444486638059144, 0.005588538643123506, 0.006976692394159365, 0.005756361016269209, 0.008318556481036826, 0.006058584267589106, 0.005958677231818661, 0.005690411649186393, 0.004592864052287821, 0.006078065245941482, 0.005857519124043935, 0.005982090150755921, 0.006352407464023115, 0.005048790435562249, 0.007742884634495495, 0.005250391752823539, 0.005444486638059144, 0.007589360227113005, 0.005490955026789583, 0.00559282803285247, 0.007745922952220178, 0.006080209940805965, 0.0058943363858842066, 0.006859091625756946, 0.005936336660313642, 0.005928115329999795, 0.005198740351503936, 0.00452101677432768, 0.005785493121511755, 0.008640081986137058, 0.005843399882852763, 0.005797646392410484, 0.0046627453599555205, 0.005702386195513083, 0.005079352337381114, 0.005605696202039361, 0.007053186510992551, 0.0050037518434081305, 0.004331390003393079, 0.0055299169834943365, 0.005161208191375504, 0.0046146684500767195, 0.006368850124650809, 0.005634113408993746, 0.005631968714129264, 0.005363524406924955, 0.004829852834813062, 0.007070522794480445, 0.004405560700789742, 0.004955674933529329, 0.005206246783529622, 0.004391977633314691, 0.005066662892766263, 0.005301328255854982, 0.004915283180248255, 0.005067377791054424, 0.005726514012738504, 0.004910815065947251, 0.007804365887277309, 0.006397982229893354, 0.0055308106063545375, 0.005562266131033605, 0.00508542897283048, 0.004883827655569188, 0.005870387293230826, 0.005919715275113908, 0.004610915234063877, 0.006737737641341683, 0.0049331556374522705, 0.004639511165590301, 0.007548789749259891, 0.005235378888772167, 0.0067002054812132505, 0.005005360364556491, 0.004507612431424668, 0.007282668861492105, 0.0050359222663753575, 0.0049767644330300665, 0.005134935679285601, 0.00560426640546304]
#
# fig_size = (4.5,4.5)
# fig, ax1 = plt.subplots(figsize=fig_size)
#
# line1, =ax1.plot(t_k, t_accuracy, label='Accuracy', linewidth=3)
# ax1.set_xlabel('Probability threshold (%)')
# # ax1.set_ylabel('Accuracy (%)')
# for t in ax1.get_yticklabels():
#     t.set_color('b')
# ax1.set_ylim([40,100])
#
# ax2 = ax1.twinx()
# line2, =ax2.plot(t_k, t_time, 'r', label = 'Processing time', linewidth=3)
# # ax2.set_ylabel('Processing time (ms)')
# plt.legend(handles=[line1,line2], loc='upper left')
# plt.locator_params(axis='y',nbins=4)
# for t in ax2.get_yticklabels():
#     t.set_color('r')
# ax2.set_ylim([0,0.04])
#
# fig.savefig('ha.pdf')
#
# plt.show()

# # #final cumu + hist
# som.set_classify_method(config['method'], **kargs)
# plt.show()
