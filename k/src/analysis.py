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

data = pd.read_csv('crash_data.csv')
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
    print(data.describe())

    print("\n")
    #выведем список уникальных имен ОТКАЗОВ СИСТЕМЫ
    names = sorted(set(data['error_name']))
    print(names)
    print("\n")
    
#def descriptiveAnalysis():
 
#def diagnosticAnalysis():

#def predictiveAnalysis():

def dataVisualization():
    plt.figure(figsize=(40, 20))
    sns.countplot(data['error_name'], label="Count")
    plt.savefig("density.png")

    data.corr().style.background_gradient(cmap='coolwarm')
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(data.corr(), annot=True)
    plt.savefig("corr.png")

    # f, subplots = plt.subplots(len(data.error_name.unique()), figsize=(12, 20))
    # for i, error_name in enumerate(data.error_name.unique()):
    #     errors = data[data.error_name == error_name]['Pid']
    #     ax = subplots[i]
    #     dist_plot = sns.distplot(errors, ax=ax)
    #     dist_plot.set_title(error_name)
    #
    # plt.tight_layout(h_pad=1)
    # plt.show()
    #
    data.hist(figsize=(15, 8), layout=(3, 6))
    plt.savefig("hist.png")

    # # Bloxpot
    # errors_name = data.error_name.unique()
    # N = len(errors_name)
    # c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N)]
    #
    # fig = go.Figure(data=[go.Box(x=errors_name,
    #                              y=data[data.error_name == error_name].Threads,
    #                              name=error_name,
    #                              marker_color=c[i]) for i, error_name in enumerate(errors_name)])
    #
    # fig.update_layout(xaxis=dict(showgrid=False,
    #                              zeroline=False,
    #                              showticklabels=True),
    #                   yaxis=dict(zeroline=False,
    #                              gridcolor='white'),
    #                   paper_bgcolor='rgb(233,233,233)',
    #                   plot_bgcolor='rgb(233,233,233)')
    #
    # fig.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="error_name",
    #                                                                     font=dict(family="Courier New, monospace",
    #                                                                               size=13,
    #                                                                               color="#7f7f7f"))),
    #                   yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Threads",
    #                                                                     font=dict(family="Courier New, monospace",
    #                                                                               size=18,
    #                                                                               color="#7f7f7f"))))
    #
    # fig.update_layout(title_text="Bloxpot Threads по error_name")
    # fig.show()



checkVersions()

transformationAndCleaning()

understandingData()

dataVisualization()
