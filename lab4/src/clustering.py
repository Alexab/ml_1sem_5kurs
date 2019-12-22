"""Для работы с большими данными"""
from math import sqrt

import numpy as np
import pandas as pd
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

    # # c нормированием
    #
    # data1 = data[data.price < 2000]
    #
    # df_knn = data1[['latitude',
    #                 'longitude',
    #                 'minimum_nights',
    #                 'number_of_reviews',
    #                 'reviews_per_month',
    #                 'calculated_host_listings_count',
    #                 'availability_365',
    #                 'price']]
    # df_knn.apply(pd.to_numeric)
    #
    # df_knn = shuffle(df_knn)
    #
    # df_norm = (df_knn[['latitude',
    #                    'longitude',
    #                    'minimum_nights',
    #                    'number_of_reviews',
    #                    'reviews_per_month',
    #                    'calculated_host_listings_count',
    #                    'availability_365',
    #                    'price']] -
    #            df_knn[['latitude',
    #                    'longitude',
    #                    'minimum_nights',
    #                    'number_of_reviews',
    #                    'reviews_per_month',
    #                    'calculated_host_listings_count',
    #                    'availability_365',
    #                    'price']].min()) / \
    #           (df_knn[['latitude',
    #                    'longitude',
    #                    'minimum_nights',
    #                    'number_of_reviews',
    #                    'reviews_per_month',
    #                    'calculated_host_listings_count',
    #                    'availability_365',
    #                    'price']].max() -
    #            df_knn[['latitude',
    #                    'longitude',
    #                    'minimum_nights',
    #                    'number_of_reviews',
    #                    'reviews_per_month',
    #                    'calculated_host_listings_count',
    #                    'availability_365',
    #                    'price']].min())
    #
    # # df_norm = pd.concat([df_norm, df_knn[['price']]], axis=1)
    #
    # df_norm = df_norm[(pd.notnull(data1['latitude'])) &
    #                   (pd.notnull(data1['longitude'])) &
    #                   (pd.notnull(data1['minimum_nights'])) &
    #                   (pd.notnull(data1['number_of_reviews'])) &
    #                   (pd.notnull(data1['reviews_per_month'])) &
    #                   (pd.notnull(data1['calculated_host_listings_count'])) &
    #                   (pd.notnull(data1['availability_365'])) &
    #                   (pd.notnull(data1['price']))]
    #
    # df_norm = df_norm.round(6)
    # df_norm = df_norm.dropna()
    # df_norm.apply(pd.to_numeric)
    #
    # from sklearn.cluster import KMeans
    #
    # clu = KMeans(n_clusters=4, random_state=0)
    #
    # clu.fit(df_norm)
    #
    # d1 = df_norm.copy()
    #
    # d1['cluster'] = clu.labels_
    #
    # d_cluster = d1.groupby('cluster').mean()
    #
    # df_cluster = d_cluster[['price', 'minimum_nights']].reset_index()
    #
    # df_cluster.plot(kind='bar', x='cluster', y='price', figsize=(12, 8), legend=False)
    # plt.title("Airbnb superhost price for KMean clusters on minimum_nights", y=1.03)
    # plt.ylabel("Price ($)", labelpad=12)
    # plt.xlabel("Cluster", labelpad=12)
    # plt.ylim(100, 150)
    # plt.xticks(rotation=360)
    # plt.show()
    #
    # from sklearn import metrics
    # from sklearn.cluster import KMeans
    # from sklearn.cluster import Birch
    # from sklearn.cluster import AgglomerativeClustering
    #
    # bestSil = -1
    # for k in range(2, 6):
    #     clus = [KMeans(n_clusters=k, n_jobs=-1), Birch(n_clusters=k), AgglomerativeClustering(n_clusters=k)]
    #     for cl in clus:
    #         res = cl.fit(df_norm)
    #         sil = metrics.silhouette_score(df_norm, res.labels_)
    #         print(str(cl)[:10] + ' with k=' + str(k) + ": " + str(round(sil, 4)))
    #         if sil > bestSil:
    #             bestSil = sil
    #             bestCl = cl
    #             bestK = k
    # print('***********************************************')
    # print('Best algorithm is... ' + str(bestCl)[:8] + '     with k=' + str(bestK))
    # print('**********************')
    # print('With Silhouette Score ' + str(bestSil))
    #
    # # data['cancellation_strict'] = data['cancellation_policy'].apply(lambda x: 'Yes' if x == 'strict' else 'No')
    # sns.catplot(y='room_type', x='price', col='minimum_nights', kind='bar', data=data)
    # plt.show()
    #
    # print(data['room_type'].value_counts() / len(data) * 100)


checkVersions()
transformationAndCleaning()
# understanding()
clusteringAnalysis()
