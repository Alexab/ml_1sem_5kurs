"""Для работы с большими данными"""
import numpy as np
import pandas as pd
import sklearn as sklearn

"""Для визуализации"""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

"""Plotly-визуализация"""
import plotly as plotly
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')


"""Загрузим датасет"""
data = pd.read_csv('AB_NYC_2019.csv')
data.tail()

def checkVersions():
    print("Numpy version: {}".format(np.version.version))
    print("Pandas version: {}".format(pd.__version__))
    print("Matplotlib version: {}".format(matplotlib.__version__))
    print("Seaborn version: {}".format(sns.__version__))
    print("Geopandas version: {}".format(gpd.__version__))
    print("Plotly version: {}".format(plotly.__version__))
    print("Scikit-Learn version: {}".format(sklearn.__version__))
    print("\n")

def transformationAndCleaning():
    # проверка на пустые ячейки
    data.isnull().sum().sort_values(ascending = False) # по убыванию

    print(data.tail())
    # для визуального анализа данных
    print(data.dropna())

    REPLACE_NAME_VALUE = "$"
    REPLACE_HOST_NAME_VALUE = "#"

    # заполним пустые (NaN) ячейки
    data['name'].fillna(REPLACE_NAME_VALUE, inplace = True)
    data['host_name'].fillna(REPLACE_HOST_NAME_VALUE, inplace = True)

    # убираем что не нужно
    data.drop(['last_review'], axis = 1, inplace = True)
    # заполняем невалидные ячейки
    data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

    print("\n")
    print("Length of Airbnb DataFrame that match with Name ="
          " \"{}\": {}".format(REPLACE_NAME_VALUE, len(data[data.name == REPLACE_NAME_VALUE])))
    print("Length of Airbnb DataFrame that match with Host_Name ="
          " \"{}\": {}".format(REPLACE_HOST_NAME_VALUE, len(data[data.host_name == REPLACE_HOST_NAME_VALUE])))
    print("\n")

    print(data.head())
    print("\n")

def understanding():
    print(data.head())

    print(data.describe())

    print(data.isnull().any().any())

    plt.rcParams['figure.figsize'] = (18, 8)

    plt.subplot(1, 2, 1)
    sns.set(style='whitegrid')
    sns.distplot(data['price'])
    plt.title('Distribution of price', fontsize=20)
    plt.xlabel('Range of price')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.set(style='whitegrid')
    sns.distplot(data['minimum_nights'], color='red')
    plt.title('Distribution of minimum_nights', fontsize=20)
    plt.xlabel('Range of minimum_nights')
    plt.ylabel('Count')
    plt.show()
    # plt.savefig("1.png")

    labels = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
    size = data['neighbourhood_group'].value_counts()
    colors = ['lightgreen', 'orange', 'blue', 'red', 'pink']
    explode = [0, 0.1, 0.2, 0.3, 0.4]

    plt.rcParams['figure.figsize'] = (9, 9)
    plt.pie(size, colors=colors, explode=explode, labels=labels, shadow=True, autopct='%.2f%%')
    plt.title('neighbourhood_group', fontsize=20)
    plt.axis('off')
    plt.legend()
    plt.show()
    # plt.savefig("2.png")

    sns.pairplot(data)
    plt.title('Pairplot for the Data', fontsize=20)
    plt.show()
    # plt.savefig("3.png")

    plt.rcParams['figure.figsize'] = (15, 8)
    sns.heatmap(data.corr(), cmap='Wistia', annot=True)
    plt.title('Heatmap for the Data', fontsize=20)
    plt.show()
    # plt.savefig("4.png")

    plt.rcParams['figure.figsize'] = (18, 7)
    sns.boxenplot(data['neighbourhood_group'], data['price'], palette='Blues')
    plt.title('neighbourhood_group vs price', fontsize=20)
    plt.show()
    # plt.savefig("5.png")

    plt.rcParams['figure.figsize'] = (18, 7)
    sns.violinplot(data['neighbourhood_group'], data['price'], palette='rainbow')
    plt.title('neighbourhood_group vs price', fontsize=20)
    plt.show()
    # plt.savefig("6.png")

    plt.rcParams['figure.figsize'] = (18, 7)
    sns.stripplot(data['neighbourhood_group'], data['price'], palette='Purples', size=10)
    plt.title('neighbourhood_group vs price', fontsize=20)
    plt.show()
    # plt.savefig("7.png")

    x = data['availability_365']
    y = data['minimum_nights']
    z = data['price']
    sns.lineplot(x, y, color='blue')
    sns.lineplot(x, z, color='pink')
    plt.title('availability_365 Income vs minimum_nights and price', fontsize=20)
    plt.show()
    # plt.savefig("8.png")

def clusteringAnalysis():
    x = data.iloc[:, [9, 10]].values
    print(x.shape)

    # K-means Algorithm

    from sklearn.cluster import KMeans

    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        km.fit(x)
        wcss.append(km.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method', fontsize=20)
    plt.xlabel('No. of Clusters')
    plt.ylabel('wcss')
    plt.show()
    # plt.savefig("9.png")

    km = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_means = km.fit_predict(x)

    plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s=100, c='pink', label='miser')
    plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s=100, c='yellow', label='general')
    plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s=100, c='cyan', label='target')
    plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s=100, c='magenta', label='spendthrift')
    plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s=100, c='orange', label='careful')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c='blue', label='centeroid')

    plt.style.use('fivethirtyeight')
    plt.title('K Means Clustering', fontsize=20)
    plt.xlabel('price')
    plt.ylabel('minimum_nights')
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("10.png")


    # 3D

    x = data[['availability_365', 'price', 'minimum_nights']].values
    km = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    labels = km.labels_
    centroids = km.cluster_centers_

    data['labels'] = labels
    trace1 = go.Scatter3d(
        x=data['availability_365'],
        y=data['price'],
        z=data['minimum_nights'],
        mode='markers',
        marker=dict(
            color=data['labels'],
            size=10,
            line=dict(
                color=data['labels'],
                width=12
            ),
            opacity=0.8
        )
    )
    df = [trace1]

    layout = go.Layout(
        title='availability_365 vs price vs minimum_nights',
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            xaxis=dict(title='availability_365'),
            yaxis=dict(title='price'),
            zaxis=dict(title='minimum_nights')
        )
    )

    fig = go.Figure(data=df, layout=layout)
    fig.show()

checkVersions()
transformationAndCleaning()
understanding()
clusteringAnalysis()
