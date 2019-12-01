from time import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graphviz import Source
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decisionTree(x_train, y_train, x_test, y_test):
    tree_array = [6, 8, 10, 12]

    crit = 'entropy'

    for i in tree_array:
        t1 = time()
        clf_tree = DecisionTreeClassifier(criterion = crit, max_depth = i, random_state = 20, presort = True)

        clf_tree.fit(X = x_train, y = y_train)

        err_train = round(np.mean(y_train != clf_tree.predict(x_train)) * 100, 4)
        err_test = round(np.mean(y_test != clf_tree.predict(x_test)) * 100, 4)

        t = -(t1 - time())

        print("Глубина дерева: {}, ошибка на обучающей: {}, ошибка на тестовой: {}, время {}".format(clf_tree.get_depth(), err_train, err_test, t))

        dotfile = export_graphviz(clf_tree,
                                  class_names=['ArrayIndexOutOfBoundsException', 'BadTokenException',
                                                'CalledFromWrongThreadException', 'ClassNotFoundException',
                                                'IllegalMonitorStateExceptio', 'IllegalStateException',
                                                'InternalError', 'NetworkOnMainThreadException',
                                                'NoClassDefFoundError', 'NullPointerException',
                                                'OutOfMemoryError', 'RuntimeException'], out_file=None,
                                                                                            filled=True, node_ids=True)
        graph = Source(dotfile)
        # Сохраним дерево как toy_example_tree_X.png, где Х - entropy или gini, критерий качестве разбиения
        graph.format = 'png'
        graph.render(str(clf_tree.get_depth())+"tree_example_tree_{}".format(crit), view=True)

#
#
#
# def plot_dt(X, y):
#     X_mat = X[['VmLck', 'VmLck']].as_matrix()
#     y_mat = y.as_matrix()
#
#     # Create color maps
#     cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
#     cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#AFAFAF'])
#
#     clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=20, presort=True)
#
#     clf_tree.fit(X=X_mat, y=y_mat)
#
#     mesh_step_size = .01
#     plot_symbol_size = 50
#
#     x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
#     y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
#
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
#                          np.arange(y_min, y_max, mesh_step_size))
#     Z = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
#     plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor='black')
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#
#     patch0 = mpatches.Patch(color='#8B0000', label='ArrayIndexOutOfBoundsException')
#     patch1 = mpatches.Patch(color='#C71585', label='BadTokenException')
#     patch2 = mpatches.Patch(color='#FF4500', label='CalledFromWrongThreadException')
#     patch3 = mpatches.Patch(color='#FFD700', label='ClassNotFoundException')
#     patch4 = mpatches.Patch(color='#6A5ACD', label='IllegalMonitorStateExceptio')
#     patch5 = mpatches.Patch(color='#8B4513', label='IllegalStateException')
#     patch6 = mpatches.Patch(color='#FF00FF', label='InternalError')
#     patch7 = mpatches.Patch(color='#00FF00', label='NetworkOnMainThreadException')
#     patch8 = mpatches.Patch(color='#000080', label='NoClassDefFoundError')
#     patch9 = mpatches.Patch(color='#FFA07A', label='NullPointerException')
#     patch10 = mpatches.Patch(color='#8FBC8F', label='OutOfMemoryError')
#     patch11 = mpatches.Patch(color='#696969', label='RuntimeException')
#
#     plt.figure(figsize=(30, 20))
#
#     plt.legend(handles=[patch0, patch1, patch2, patch3, patch4, patch5,
#                         patch6, patch7, patch8, patch9, patch10, patch11])
#
#     plt.xlabel('1')
#     plt.ylabel('2')
#     plt.title("12-Class classification")
#     plt.show()

if __name__ == '__main__':

    dataset = pd.read_csv('crash_data_1.csv')

    del dataset['State']
    del dataset['TracerPid']
    del dataset['FDSize']
    del dataset['VmPin']
    del dataset['VmStk']
    del dataset['VmExe']
    del dataset['SigPnd']
    del dataset['ShdPnd']
    del dataset['SigBlk']
    del dataset['SigIgn']
    del dataset['CapInh']
    del dataset['CapPrm']
    del dataset['CapEff']
    del dataset['CapBnd']
    del dataset['Seccomp']
    del dataset['Cpus_allowed']
    del dataset['Cpus_allowed_list']
    del dataset['n']
    del dataset['crash']

    objects = [c for c in dataset.columns if dataset[c].dtype.name == 'object']

    dataset_describe = dataset.describe(include=[object])

    for c in objects:
        dataset[c] = dataset[c].fillna(dataset_describe[c]['top'])

    dataset.count(axis=0)

    binary = [c for c in objects if dataset_describe[c]['unique'] == 2]
    nonBinary = [c for c in objects if dataset_describe[c]['unique'] > 2]
    nonBinary3 = [c for c in objects if dataset_describe[c]['unique'] <= 3]

    dataset_nonBinary = pd.get_dummies(dataset[nonBinary3])

    print("\n")

    dataset_numerical = dataset[['Tgid', 'Pid', 'PPid', 'VmPeak', 'VmSize', 'VmLck',
                                 'VmHWM', 'VmRSS', 'VmData', 'VmLib', 'VmPTE', 'VmSwap', 'Threads',
                                 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches']]

    dataset_numerical = (dataset_numerical - dataset_numerical.mean()) / dataset_numerical.std()

    temp = pd.concat((dataset_numerical, dataset[binary], dataset_nonBinary, dataset[['error_name']]), axis=1)

    dataset = pd.DataFrame(temp)

    Xn = dataset.drop(('error_name'), axis=1)
    Yn = dataset['error_name']

    train_data, test_data, train_labels_out, test_labels_out = train_test_split(Xn, Yn, test_size=0.3, random_state=11)

    print("\n")

    decisionTree(train_data, train_labels_out, test_data, test_labels_out)

    print("\n")

    # names = sorted(set(dataset1['crash']))
    # print(names)
    #
    # X = dataset1[['Tgid', 'Pid', 'PPid', 'VmPeak', 'VmSize', 'VmLck',
    #              'VmHWM', 'VmRSS', 'VmData', 'VmLib', 'VmPTE', 'VmSwap', 'Threads',
    #              'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches']]
    # y = dataset1['crash']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #
    # plot_dt(X_train, y_train)






