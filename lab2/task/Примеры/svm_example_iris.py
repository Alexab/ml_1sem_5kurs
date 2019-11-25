# FROM: https://pythonprogramming.net/predictions-svm-machine-learning-tutorial/

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b', 9: 'c', 11: 'm'}

        plt.ion()

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data_train, labels_train):
        self.data_train = data_train
        self.labels_train = labels_train
        self.N = data_train.shape[0]

        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        self.max_feature_value = np.amax(data_train)
        self.min_feature_value = np.amin(data_train)

        # support vectors yi(xi.w+b) = 1

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]

        # extremely expensive
        b_range_multiple = 2
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        #
                        # #### add a break here later..
                        for n in range(self.N):
                            # we want  yi*(np.dot(w_t,xi)+b) >= 1 for correct classification
                            if labels_train[n] * (np.dot(w_t, data_train[n]) + b) < 1:
                                found_option = False
                                    # print(xi,':',yi*(np.dot(w_t,xi)+b))

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

            self.visualize()

        for n in range(self.N):
            yi = labels_train[n]
            print(data_train[n], ':', yi * (np.dot(self.w, data_train[n]) + self.b))

    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification+10], edgecolor='b')
        return classification

    def visualize(self):

        plt.cla()

        [self.ax.scatter(self.data_train[i][0], self.data_train[i][1], s=100, color=self.colors[self.labels_train[i]], edgecolor='b') for i in range(self.N)]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.draw()
        plt.pause(3)





# ------------------------------------------------------------------------------------------------------------------- #
# Исходные параметры
# ------------------------------------------------------------------------------------------------------------------- #
# Наша игрушечная база в формате pandas DataFrame
data = load_iris()
labels = data.target
samples = data.data
print("Dataset Iris:\n\tx_data shape: {}\n\tfeatures: {}".format(samples.shape, data.feature_names))

# ------------------------------------------------------------------------------------------------------------------- #
# 1. Выберем два первых признака и приравняем третий класс к первому
# ------------------------------------------------------------------------------------------------------------------- #
y, y_names  = labels, data.target_names
y2 = np.copy(y)
y2[y2 == 2] = 1
y2[y2 == 0] = -1
y2_names = np.copy(y_names[:2])

# ------------------------------------------------------------------------------------------------------------------- #
# 2. Разделим базу на обучающую и тестовую выборку
# ------------------------------------------------------------------------------------------------------------------- #

data_train, data_test, labels_train, labels_test = train_test_split(samples[:, 1:3], y2, test_size = 0.1, random_state=20)

svm = Support_Vector_Machine()
svm.fit(data_train = data_train, labels_train = labels_train)

preds = []
for p in data_test:
    preds.append(svm.predict(p))
print(accuracy_score(labels_test, np.array(preds)))

svm.visualize()


