"""Для работы с большими данными"""
import numpy as np
import pandas as pd

"""Для визуализации"""
import matplotlib
import seaborn as sns
import geopandas as gpd
from matplotlib import pyplot as plt

"""Plotly-визуализация"""
import plotly as plotly
import plotly.graph_objects as go

"""Машинное обучение"""
import sklearn

"""Загрузим датасет"""

data = pd.read_csv('new_crash_data.csv')
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

    print(data.tail())
    # для визуального анализа данных
    print(data.dropna())

    # убираем что не нужно
    data.drop(['State', 'TracerPid', 'FDSize', 'VmPin', 'VmStk',
                 'VmExe', 'SigPnd', 'ShdPnd', 'SigBlk', 'SigIgn',
                 'SigCgt', 'CapInh', 'CapPrm', 'CapEff', 'CapBnd',
                 'Seccomp', 'Cpus_allowed', 'Cpus_allowed_list', 'n'], axis = 1, inplace = True)


    print("\n")
    print("Length of CRASHES DataFrame =", len(data['error_name']))

    print(data.head())
    print("\n")

def understandingData():
    # выведем количество записей в базе
    print(len(data.index))

    # выведем количество (строки, столбцы)
    print(data.shape)
    print("\n")

    # первая строка, раскиданная по столбцам
    print(data.loc[0])
    print("\n")

    # типы по стообцам
    print(data.info(verbose = True))
    print("\n")

    # понимание по значениям
    print(data.describe(exclude=[np.object]))

    print("\n")
    #выведем список уникальных имен ОТКАЗОВ СИСТЕМЫ
    names = sorted(set(data['error_name']))
    print(names)
    print("\n")

def dataVisualization():
    plt.figure(figsize=(40, 20))
    sns.countplot(data['error_name'], label="Count")
    plt.savefig("density.png")

    data.corr().style.background_gradient(cmap='coolwarm')
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(data.corr(), annot=True)
    plt.savefig("corr.png")

    data.hist(figsize=(15, 8), layout=(3, 6))
    plt.savefig("hist.png")


checkVersions()

transformationAndCleaning()

understandingData()

dataVisualization()
