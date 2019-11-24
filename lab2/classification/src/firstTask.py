# created by БорискинМА
# 01.11.19,10.11.19,14.11.19, 23.11.2019
# PyCharm 2019.3 Professional (JetBrains Product Pack for Students)
import random
import warnings
from itertools import islice
from time import time

import folium
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap
from matplotlib import pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
data = pd.read_csv('AB_NYC_2019.csv')


def totalABNYC2019():
    print('Количество записей в базе:', len(set(data['id'])), '\n')


def histogram():
    data.info()
    print('\n')

    data.hist(figsize = (15, 8), layout = (2, 5))

    plt.show()


def areThereAnyMissingItems():
    print(data.isnull().sum())
    print('\n')
    print('Broken items in', 'столбцы \"дата последнего отзыва\" и \"количество отзывов в месяц\"', 'are the same?',
          (data['last_review'].isnull() == data['reviews_per_month'].isnull()).all())
    print('\n')


def fillMissingItems():
    data.loc[data['reviews_per_month'].isnull(), 'reviews_per_month'] = 0


def checkNumericVariables():
    print(data.describe())
    print('\n')


def removeUnnecessaryColumns():
    data.drop(['id', 'name', 'host_name', 'last_review'], axis = 1, inplace = True)


def cleanAbnormalities():
    dataNYC = data[data['price'] > 0]
    dataNYC = data[data['minimum_nights'] < 366]
    dataNYC.reset_index(drop = True, inplace = True)
    dataNYC.info()
    print('\n')
    return dataNYC


def latlonNbrhdGr():
    names_neigh = sorted(set(data['neighbourhood_group']))

    colors = []
    for i in range(len(names_neigh)):
        color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        colors.append(color)

    color_dict = dict()

    for color, name in zip(colors, names_neigh):
        color_dict.update({name: color})

    data_colors = data['neighbourhood_group'].map(color_dict)
    ny_plot = data.plot(kind = 'scatter', x = 'latitude', y = 'longitude', color = data_colors)
    plt.show()


def heatMap():
    m = folium.Map([40.7128, -74.0060], zoom_start = 11)
    data_loc = data[['latitude', 'longitude']].values
    data_loc = data_loc.tolist()
    HeatMap(data[['latitude', 'longitude']].dropna(), radius = 8,
            gradient = {0.2: 'blue', 0.4: 'purple', 0.6: 'orange', 1.0: 'red'}).add_to(m)

    # if you want you can open it manually in your webbrowser
    m.save("heatMap.html")


def correlation():
    print(set(data['neighbourhood_group']))
    print('\n')

    for group in set(data['neighbourhood_group']):
        print(group)
        print(data[data['neighbourhood_group'] == group][['price', 'minimum_nights', 'availability_365']].corr())
        print('\n')
    sns.distplot(data['minimum_nights'])
    plt.show()

    corr = data[data['neighbourhood_group'] == group][['price', 'minimum_nights', 'availability_365']].corr()
    sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)
    plt.title('Correlation Color Matrix')
    plt.show()

    print('\n')


def latlonPrice():
    all_prices = data['price'].values

    price_colors = cm.rainbow(np.linspace(0, 1, len(all_prices)))

    price_plot = data.plot(kind = 'scatter', x = 'latitude', y = 'longitude', color = price_colors)
    plt.show()


def wordsCounter():
    name = pd.read_csv('AB_NYC_2019.csv', usecols = [1], squeeze = True)

    dictionary = {}
    dictionary = name.str.lower().str.split(None, expand = True).stack().value_counts()

    print('25 наиболее часто встречаемых слов в колонке name:\n')
    print(list(islice(dictionary.items(), 25)))
    print('\n')


# 1
totalABNYC2019()
# 2
histogram()
# 3
areThereAnyMissingItems()
fillMissingItems()
checkNumericVariables()
removeUnnecessaryColumns()
# noinspection PyRedeclaration
data = cleanAbnormalities()
# 4
latlonNbrhdGr()
heatMap()
# 5
correlation()
# 6
latlonPrice()
# 7
wordsCounter()


def classificationsWorker(price):
    dataset = pd.read_csv('AB_NYC_2019.csv', parse_dates = ['last_review'], index_col = ['id'])

    objects = [c for c in dataset.columns if dataset[c].dtype.name == 'object']

    dataset['reviews_per_month'] = dataset['reviews_per_month'].fillna(0)

    if price:
        dataset['price'] = pd.qcut(dataset['price'], 9, labels = list(map(str, np.arange(9))))
    else:
        del dataset['last_review']

    dataset = dataset.dropna(axis = 0, subset = ['name', 'host_name'])

    dataset_describe = dataset.describe(include = [object])

    for c in objects:
        dataset[c] = dataset[c].fillna(dataset_describe[c]['top'])

    dataset.count(axis = 0)

    binary = [c for c in objects if dataset_describe[c]['unique'] == 2]
    nonBinary = [c for c in objects if dataset_describe[c]['unique'] > 2]
    nonBinary3 = [c for c in objects if dataset_describe[c]['unique'] <= 3]

    dataset_nonBinary = pd.get_dummies(dataset[nonBinary3])

    print("\n")

    if price:
        dataset_pr_class = dataset[['price']]
        dataset_geo = dataset[['latitude', 'longitude']]

        dataset_numerical = dataset[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]
        dataset_numerical = (dataset_numerical - dataset_numerical.mean()) / dataset_numerical.std()
        dataset_geo = (dataset_geo - dataset_geo.mean()) / dataset_geo.std()

        temp = pd.concat((dataset_numerical, dataset[binary], dataset_nonBinary, dataset_pr_class, dataset_geo), axis = 1)

        dataset = pd.DataFrame(temp)

        Xn = dataset.drop(('price'), axis = 1)
        Yn = dataset['price']
        Yn = Yn.astype('int')

        x_train, x_test, y_train, y_test = train_test_split(Xn, Yn, test_size = 0.2, random_state = 11)

        print("Предсказание по цене")

        print(kNN(x_train, y_train, x_test, y_test)[0])
        print(kNN(x_train, y_train, x_test, y_test)[1])

        print("\n")

        decisionTree(x_train, y_train, x_test, y_test, price)

        print("\n")

        print("Наивный Байес по цене:")
        print(naiveBayes(x_train, y_train, x_test, y_test))

        print("\n")

        print("SVC по цене:")
        print(supportVectorMachine(x_train, y_train, x_test, y_test))

    else:
        dataset_numerical = dataset[
            ['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
             'calculated_host_listings_count', 'availability_365']]
        dataset_numerical = (dataset_numerical - dataset_numerical.mean()) / dataset_numerical.std()

        temp = pd.concat((dataset_numerical, dataset[binary], dataset_nonBinary, dataset[['neighbourhood_group']]),
                         axis=1)

        dataset = pd.DataFrame(temp)

        Xn = dataset.drop(('neighbourhood_group'), axis=1)
        Yn = dataset['neighbourhood_group']

        x_train, x_test, y_train, y_test = train_test_split(Xn, Yn, test_size=0.2, random_state=11)

        print("Предсказание по району")

        print(kNN(x_train, y_train, x_test, y_test)[0])
        print(kNN(x_train, y_train, x_test, y_test)[1])

        print("\n")

        decisionTree(x_train, y_train, x_test, y_test, price)

        print("\n")

        print("Наивный Байес по району:")
        print(naiveBayes(x_train, y_train, x_test, y_test))

        print("\n")

        print("SVC по району:")
        print(supportVectorMachine(x_train, y_train, x_test, y_test))


def kNN(x_train, y_train, x_test, y_test):
    knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
    knn.fit(x_train, y_train)

    n_neighbors_array = [1, 3, 5, 7, 10, 15]

    # https://iq.opengenus.org/euclidean-vs-manhattan-vs-chebyshev-distance/
    metrics = ['euclidean', 'manhattan', 'chebyshev']

    err = {'euclidean': [], 'manhattan': [], 'chebyshev': []}
    t = {'euclidean': [], 'manhattan': [], 'chebyshev': []}

    for i in n_neighbors_array:
        for met in metrics:
            t0 = time()

            knn = neighbors.KNeighborsClassifier(n_neighbors = i, metric = met, weights = 'distance')
            knn.fit(x_train, y_train)

            y_test_predict = knn.predict(x_test)

            t1 = time()

            err[met].append(round(np.mean(y_test != y_test_predict) * 100))

            t[met].append(round(t1 - t0, 2))

    knn_time = pd.DataFrame(t, index = n_neighbors_array)
    knn_err = pd.DataFrame(err, index = n_neighbors_array)

    return knn_err, knn_time

def decisionTree(x_train, y_train, x_test, y_test, integer):
    tree_array = [6, 10, 14, 18, 22, 26]

    crit = 'entropy'

    for i in tree_array:
        t1 = time()
        clf_tree = DecisionTreeClassifier(criterion = crit, max_depth = i, random_state = 20, presort = True)

        clf_tree.fit(X = x_train, y = y_train)

        if integer:
            err_train = round(np.mean(y_train != clf_tree.predict(x_train).astype('int')) * 100, 2)
            err_test = round(np.mean(y_test != clf_tree.predict(x_test).astype('int')) * 100, 2)
        else:
            err_train = round(np.mean(y_train != clf_tree.predict(x_train)) * 100, 2)
            err_test = round(np.mean(y_test != clf_tree.predict(x_test)) * 100, 2)

        t = t1 - time()

        print("Глубина дерева: {}, ошибка на обучающей: {}, ошибка на тестовой: {}, время {}".format(clf_tree.get_depth(), err_train, err_test, t))


def naiveBayes(x_train, y_train, x_test, y_test):
    model = GaussianNB()

    model.fit(x_train, y_train)

    return testTrain(x_train, y_train, x_test, y_test, model)


def supportVectorMachine(x_train, y_train, x_test, y_test):
    svc = SVC(gamma = 'scale')

    svc.fit(x_train, y_train)

    return testTrain(x_train, y_train, x_test, y_test, svc)

def testTrain(x_train, y_train, x_test, y_test, model):
    err_train = round(np.mean(y_train != model.predict(x_train)) * 100, 2)
    err_test = round(np.mean(y_test != model.predict(x_test)) * 100, 2)

    return err_train, err_test


classificationsWorker(False)
classificationsWorker(True)
