"""Для работы с большими данными"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

"""Для визуализации"""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

"""Plotly-визуализация"""
import plotly as plotly
import plotly.express as px
import plotly.graph_objects as go

"""Folium-визуализация"""
import folium
import folium.plugins

"""WordCloud"""
import wordcloud
from wordcloud import WordCloud

"""Машинное обучение"""
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, LinearRegression
import xgboost as xgb
from math import sqrt

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
    print("Folium version: {}".format(folium.__version__))
    print("WordCloud version: {}".format(wordcloud.__version__))
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

def descriptiveAnalysis():
    data.room_type.value_counts()

    # визуализация показывает, что люди на Airbnb в Нью-Йорке предпочитают аренду целого дома
    room_type_plot = sns.countplot(x = "room_type", order = data.room_type.value_counts().index, data = data)
    room_type_plot.set(xlabel = 'Типы размещения', ylabel = '', title = 'Распределение по типам размещения')
    for bar in room_type_plot.patches:
        h = bar.get_height()
        room_type_plot.text(
            bar.get_x() + bar.get_width() / 2.,  # bar index (x coordinate of text)
            h,  # y coordinate of text
            '%d' % int(h),  # y label
            ha = 'center',
            va = 'bottom',
            color = 'black',
            fontweight = 'bold',
            size = 14)
    plt.show()

    # посмотрим по большим районам
    data.neighbourhood_group.value_counts(dropna = False, normalize = True)

    # круговая (секторная) диаграма по района
    labels = data.neighbourhood_group.value_counts().index
    sizes = data.neighbourhood_group.value_counts().values

    explode = (0.1, 0.2, 0.3, 0.4, 0.6)

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes,
                                      explode = explode,
                                      labels = labels,
                                      autopct = '%1.1f%%',
                                      shadow = True,
                                      startangle = 90)

    ax.axis('equal')
    ax.set(title = "Где больше всего предложений аренды?")
    ax.legend(wedges,
              labels,
              title = "Neighbourhood Groups",
              loc = "center left",
              bbox_to_anchor = (1, 0, 0.5, 1))
    plt.setp(autotexts, size = 8, weight = "bold")
    plt.show()

    # районы по большим районам
    for neighbourhood_group in data.neighbourhood_group.unique():
        neighbourhoods = data.neighbourhood[data.neighbourhood_group == neighbourhood_group].unique()
        print("{} -> {} района(-ов): {}... \n".format(neighbourhood_group, len(neighbourhoods), neighbourhoods[:5]))

    sns.set(style = "white", context = "talk")
    f, subplots = plt.subplots(len(data.neighbourhood_group.unique()), 1, figsize = (20, 12))

    for i, neighbourhood_group in enumerate(data.neighbourhood_group.unique()):
        neighbourhoods = data.neighbourhood[data.neighbourhood_group == neighbourhood_group]
        ax = subplots[i]
        x = np.array(neighbourhoods.value_counts().index)
        y = neighbourhoods.value_counts().values
        sns.barplot(x = x, y = y, palette = "rocket", ax = ax)
        ax.axhline(0, color = "k", clip_on = False)
        ax.set_ylabel(neighbourhood_group)

    sns.despine(bottom = True)
    f.suptitle('Большой район по маленьким')
    plt.setp(f.axes, yticks = [])
    plt.tight_layout(h_pad = 2)
    plt.show()
    
    # плотность, распределение цен по большим районам
    sns.set(style = "whitegrid")
    BORDER_PRICE_VALUE = 1000

    sub_airbnb_price = data[data.price < BORDER_PRICE_VALUE]

    fig, ax = plt.subplots(figsize = (12, 12))
    density_neighbourhood_price_plot = sns.violinplot(ax = ax,
                                                      x = "neighbourhood_group",
                                                      y = "price",
                                                      hue = "neighbourhood_group",
                                                      data = sub_airbnb_price,
                                                      palette = "muted",
                                                      dodge = False)

    density_neighbourhood_price_plot.set(xlabel = 'Neighbourhood Group',
                                         ylabel = 'Price ($)',
                                         title = 'Плотность и распределение цен по большим районам')

    ylabels = ['${}'.format(x) for x in density_neighbourhood_price_plot.get_yticks()]

    density_neighbourhood_price_plot.set_yticklabels(ylabels)

    plt.show()

    # продвинутое соотнесение минимального количества ночей с типом размещения
    MINIMUM_NIGHTS_BORDER = 15

    sub_airbnb = data[data.minimum_nights < MINIMUM_NIGHTS_BORDER]

    fig, ax = plt.subplots(figsize = (12, 12))
    sns.set(style = "ticks", palette = "pastel")

    nights_per_room = sns.boxplot(x = "room_type",
                                  y = "minimum_nights",
                                  ax = ax,
                                  hue = "room_type",
                                  dodge = False,
                                  linewidth = 2.5,
                                  data = sub_airbnb)

    nights_per_room.set(xlabel = "Minimum nights",
                        ylabel = "Room Type",
                        title = "Минимальное бронирование для типа размещения")

    plt.legend(loc = 'upper right')
    plt.show()

    # распределение цены по большим районам
    f, subplots = plt.subplots(len(data.neighbourhood_group.unique()), figsize = (12, 20))

    for i, neighbourhood_group in enumerate(data.neighbourhood_group.unique()):
        neighbourhoods = data[data.neighbourhood_group == neighbourhood_group]['price']
        ax = subplots[i]
        dist_plot = sns.distplot(neighbourhoods, ax = ax)
        dist_plot.set_title(neighbourhood_group)

    plt.tight_layout(h_pad = 1)
    plt.show()

    # количество рецензий по имени выложившего объявление
    serie_airbnb = data.groupby("host_id")["number_of_reviews"].agg("sum")
    frame = {'host_id': serie_airbnb.index, 'number_of_reviews': serie_airbnb.values}
    df_data = pd.DataFrame(frame).sort_values('number_of_reviews', ascending = False).head(50)

    f, ax = plt.subplots(figsize = (12, 12))
    sns.barplot(x = "number_of_reviews",
                y = "host_id",
                data = df_data,
                color = "b",
                ax = ax,
                orient = "h")

    plt.show()

    top_10_host_id = df_data.host_id.unique()[:10]

    names = data.host_name[data.host_id.isin(top_10_host_id)].unique()

    print("\n")
    print(names)
    print("\n")

def makeWordCloudImage(text, colormap = "viridis", imageUrl = None):
    from PIL import Image

    if imageUrl is not None:
        nyc_mask = np.array(Image.open(imageUrl))
        wc = WordCloud(background_color = "white",
                       colormap = colormap,
                       mask = nyc_mask,
                       contour_width = 1.5,
                       contour_color = 'steelblue')
    else:
        wc = WordCloud(background_color = "white",
                       width = 1920,
                       height = 1080,
                       max_font_size = 200,
                       max_words = 200,
                       colormap = colormap)
    wc.generate(text)

    f, ax = plt.subplots(figsize = (12, 12))
    plt.imshow(wc, interpolation = "bilinear")
    plt.axis("off")
    plt.show()

def diagnosticAnalysis():
    data.corr().style.background_gradient(cmap = 'coolwarm')

    plt.figure(figsize = (12, 12))
    ax = sns.heatmap(data.corr(), annot = True)
    plt.show()

    # обработка естественного языка (NLP)
    neighbourhood_text = " ".join([neighbourhood for neighbourhood in data["neighbourhood"]])
    makeWordCloudImage(neighbourhood_text, imageUrl = "nyc_.png")

    names_text = " ".join([host_name for host_name in data["host_name"]])
    makeWordCloudImage(names_text, colormap = "RdYlGn")

def predictiveAnalysis():
    # гистограммы
    # availability_365 сортированная по большим районам и ранжированная по цене

    AIRBNB_LIMIT_SIZE = 5000

    sub_airbnb = data[:AIRBNB_LIMIT_SIZE].sort_values("price")

    fig = px.bar(sub_airbnb, x = "price",
                 y = "neighbourhood_group",
                 color = "availability_365",
                 orientation = 'h',
                 hover_data = ["host_name", "minimum_nights"],
                 height = 400,
                 title = '365 Availability Study | Price')

    fig.update_layout(xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Price",
                                                                            font = dict(family = "Courier New, monospace",
                                                                                        size = 18,
                                                                                        color = "#7f7f7f"))),
                      yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "Neighbourhood Group",
                                                                            font = dict(family = "Courier New, monospace",
                                                                                        size = 18,
                                                                                        color = "#7f7f7f"))))

    # открывается в браузере
    fig.show()

    # number_of_reviews/availability_365 для host_id/host_name соответственно
    fig = px.scatter(data, x = "availability_365",
                     y = "host_id",
                     size = "number_of_reviews",
                     color = "neighbourhood_group",
                     hover_name = "host_name",
                     title = "Number of reviews/Availability 365 per Host ID/ Host Name")

    fig.update_layout(legend_orientation="h")

    fig.update_layout(xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Availability 365",
                                                                            font = dict(family = "Courier New, monospace",
                                                                                        size = 13,
                                                                                        color = "#7f7f7f"))),
                      yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "Host Id",
                                                                            font = dict(family = "Courier New, monospace",
                                                                                        size = 18,
                                                                                        color = "#7f7f7f"))))

    fig.show()

    # Bloxpot
    neighbourhoods = data.neighbourhood.unique()
    N = len(neighbourhoods)
    c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N)]

    fig = go.Figure(data = [go.Box(x = neighbourhoods,
                                   y = data[data.neighbourhood == neighbourhood].number_of_reviews,
                                   name = neighbourhood,
                                   marker_color = c[i]) for i, neighbourhood in enumerate(neighbourhoods)])

    fig.update_layout(xaxis = dict(showgrid = False,
                                   zeroline = False,
                                   showticklabels = True),
                      yaxis = dict(zeroline = False,
                                   gridcolor = 'white'),
                      paper_bgcolor = 'rgb(233,233,233)',
                      plot_bgcolor = 'rgb(233,233,233)')

    fig.update_layout(xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Neighbourhood",
                                                                            font = dict(family = "Courier New, monospace",
                                                                                        size = 13,
                                                                                        color = "#7f7f7f"))),
                      yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "Number of Reviews",
                                                                            font = dict(family = "Courier New, monospace",
                                                                                        size = 18,
                                                                                        color = "#7f7f7f"))))

    fig.update_layout(title_text="Bloxpot Количество просмотров по району")
    fig.show()

def dataVisualization():
    # анализ по большим районам, наибольшее количетсво предложений - на Манхеттене
    fig, ax = plt.subplots(figsize = (12, 12))

    img = plt.imread('nyc_.png', 0)
    coordinates_to_extent = (-74.258, -73.7, 40.49, 40.92)
    ax.imshow(img, zorder = 0, extent = coordinates_to_extent)

    scatter_map = sns.scatterplot(x = 'longitude',
                                  y = 'latitude',
                                  hue = 'neighbourhood_group',
                                  s = 20,
                                  ax = ax,
                                  data = data)

    ax.grid(True)
    plt.legend(title = 'Neighbourhood Groups')
    plt.show()

    # обзор по цене, самая высокая на Манхеттене
    BORDER_PRICE_VALUE = 400
    sub_airbnb = data[data.price < BORDER_PRICE_VALUE]

    fig, ax = plt.subplots(figsize = (12, 12))
    cmap = plt.get_cmap('jet')
    c = sub_airbnb.price
    alpha = 0.5
    label = "airbnb"
    price_heatmap = ax.scatter(sub_airbnb.longitude,
                               sub_airbnb.latitude,
                               label = label,
                               c = c,
                               cmap = cmap,
                               alpha = 0.4)

    plt.title("Heatmap по цене ($)")
    plt.colorbar(price_heatmap)
    plt.grid(True)

    plt.show()

    # распрееление по Нью-Йорку
    heat_map = folium.Map([40.7128, -74.0060], zoom_start = 11)

    folium.plugins.HeatMap(data[['latitude', 'longitude']].dropna(),
                           radius = 8,
                           gradient = {0.2:'blue', 0.4:'purple', 0.6:'orange', 1.0:'red'}).add_to(heat_map)

    heat_map.save("heatMap.html")

checkVersions()

transformationAndCleaning()

understandingData()

descriptiveAnalysis()
diagnosticAnalysis()
predictiveAnalysis()

dataVisualization()


# вторая часть работы - регрессия (сначала kNRegressor, затем линейные регрессоры)
def regressionWorker(label1, label2):
    BORDER_PRICE_VALUE = 1000

    data1 = data[data.price < BORDER_PRICE_VALUE]

    df_knn = data1[['latitude',
                   'longitude',
                   'minimum_nights',
                   'number_of_reviews',
                   label2,
                   'calculated_host_listings_count',
                   'availability_365',
                   label1]]
    df_knn.apply(pd.to_numeric)

    df_knn = shuffle(df_knn)

    df_norm = (df_knn[['latitude',
                       'longitude',
                       'minimum_nights',
                       'number_of_reviews',
                       label2,
                       'calculated_host_listings_count',
                       'availability_365']] -
               df_knn[['latitude',
                        'longitude',
                        'minimum_nights',
                        'number_of_reviews',
                        label2,
                        'calculated_host_listings_count',
                        'availability_365']].min()) / \
              (df_knn[['latitude',
                        'longitude',
                        'minimum_nights',
                        'number_of_reviews',
                        label2,
                        'calculated_host_listings_count',
                        'availability_365']].max() -
               df_knn[['latitude',
                        'longitude',
                        'minimum_nights',
                        'number_of_reviews',
                        label2,
                        'calculated_host_listings_count',
                        'availability_365']].min())

    df_norm = pd.concat([df_norm, df_knn[[label1]]], axis=1)

    df_norm = df_norm[(pd.notnull(data1['latitude'])) &
                      (pd.notnull(data1['longitude'])) &
                      (pd.notnull(data1['minimum_nights'])) &
                      (pd.notnull(data1['number_of_reviews'])) &
                      (pd.notnull(data1[label2])) &
                      (pd.notnull(data1['calculated_host_listings_count'])) &
                      (pd.notnull(data1['availability_365'])) &
                      (pd.notnull(data1[label1]))]

    df_norm = df_norm.round(6)
    df_norm = df_norm.dropna()
    df_norm.apply(pd.to_numeric)

    x_train, x_test, y_train, y_test = train_test_split(df_norm[['latitude',
                                                                 'longitude',
                                                                 'minimum_nights',
                                                                 'number_of_reviews',
                                                                 label2,
                                                                 'calculated_host_listings_count',
                                                                 'availability_365']], df_norm[label1], test_size = 0.2, random_state = 42)

    # print(len(x_train))
    # print(len(x_test))
    #
    # print(len(y_train))
    # print(len(y_test))

    x_train = x_train[(pd.notnull(data1['latitude'])) &
                      (pd.notnull(data1['longitude'])) &
                      (pd.notnull(data1['minimum_nights'])) &
                      (pd.notnull(data1['number_of_reviews'])) &
                      (pd.notnull(data1[label2])) &
                      (pd.notnull(data1['calculated_host_listings_count'])) &
                      (pd.notnull(data1['availability_365']))]

    x_train = x_train.dropna()
    x_train = x_train.round(6)
    x_train.apply(pd.to_numeric)

    x_test = x_test[(pd.notnull(data1['latitude'])) &
                    (pd.notnull(data1['longitude'])) &
                    (pd.notnull(data1['minimum_nights'])) &
                    (pd.notnull(data1['number_of_reviews'])) &
                    (pd.notnull(data1[label2])) &
                    (pd.notnull(data1['calculated_host_listings_count'])) &
                    (pd.notnull(data1['availability_365']))]

    x_test = x_test.dropna()
    x_test = x_test.round(6)
    x_test.apply(pd.to_numeric)

    y_train.index.name = label1
    y_test.index.name = label1

    y_train = y_train[(pd.notnull(data1[label1]))]

    y_train = y_train.dropna()
    y_train = y_train.round(6)
    y_train.apply(pd.to_numeric)

    y_test = y_test[(pd.notnull(data1[label1]))]

    y_test = y_test.dropna()
    y_test = y_test.round(6)
    y_test.apply(pd.to_numeric)

    # ПОЕХАЛИ

    kNNRegression(x_train, x_test, y_train, y_test, label1)
    print("\n")
    bayesianRegression(x_train, x_test, y_train, y_test, label1)
    print("\n")
    ridgeRegression(x_train, x_test, y_train, y_test, label1)
    print("\n")
    linearRegression(x_train, x_test, y_train, y_test, label1)
    print("\n")
    LARSLassoRegression(x_train, x_test, y_train, y_test, label1)
    print("\n")

    if label1 == 'price':
        # advanced Regressions

        # закомменченные требует много времени для обучения

        # ~50 минут - 1,5 часа
        # gradientBoostingRegression(x_train, x_test, y_train, y_test)
        # print("\n")
        # ~1 час 40 минут - 3 часа
        # XGBRegression(x_train, x_test, y_train, y_test)
        # print("\n")

        LassoRegressionAdvanced(x_train, x_test, y_train, y_test)
        print("\n")
        linearRegressionAdvanced(x_train, x_test, y_train, y_test)
        print("\n")


def kNNRegression(x_train, x_test, y_train, y_test, label1):
    n_neighbors_array = [1, 3, 5, 7, 10, 15]

    for i in n_neighbors_array:
        knn = KNeighborsRegressor(n_neighbors = i,
                                  weights = 'uniform',
                                  algorithm = 'auto',
                                  leaf_size = 30, p = 2,
                                  metric = 'minkowski',
                                  metric_params = None,
                                  n_jobs = None)
        knn.fit(x_train, y_train)

        predictions = knn.predict(x_test)
        predictions = pd.DataFrame({label1: predictions})

        printResults(predictions, y_test, "Для KNeighborsRegressor:", i)

def bayesianRegression(x_train, x_test, y_train, y_test, label1):
    reg = linear_model.BayesianRidge(alpha_1 = 1e-06,
                                     alpha_2 = 1e-06,
                                     compute_score = False,
                                     copy_X = True,
                                     fit_intercept = True,
                                     lambda_1 = 1e-06,
                                     lambda_2 = 1e-06,
                                     n_iter = 300,
                                     normalize = False,
                                     tol = 0.001,
                                     verbose = False)

    reg.fit(x_train, y_train)

    predictions = reg.predict(x_test)
    predictions = pd.DataFrame({label1: predictions})

    printResults(predictions, y_test, "Для linear_model.BayesianRidge:")

def ridgeRegression(x_train, x_test, y_train, y_test, label1):
    reg = linear_model.Ridge(alpha = 0.5,
                             copy_X = True,
                             fit_intercept = True,
                             max_iter = None,
                             normalize = False,
                             random_state = None,
                             solver = 'auto',
                             tol = 0.001)

    reg.fit(x_train, y_train)

    predictions = reg.predict(x_test)
    predictions = pd.DataFrame({label1: predictions})

    printResults(predictions, y_test, "Для linear_model.Ridge:")

def linearRegression(x_train, x_test, y_train, y_test, label1):
    reg = linear_model.LinearRegression(fit_intercept = True,
                                        normalize = False,
                                        copy_X = True,
                                        n_jobs = None)

    reg.fit(x_train, y_train)

    predictions = reg.predict(x_test)
    predictions = pd.DataFrame({label1: predictions})

    printResults(predictions, y_test, "Для linear_model.LinearRegression:")

def LARSLassoRegression(x_train, x_test, y_train, y_test, label1):
    reg = linear_model.LassoLars(alpha = 0.1,
                                 copy_X = True,
                                 fit_intercept = True,
                                 fit_path = True,
                                 max_iter = 500,
                                 normalize = True,
                                 positive = False,
                                 precompute = 'auto',
                                 verbose = False)

    reg.fit(x_train, y_train)

    predictions = reg.predict(x_test)
    predictions = pd.DataFrame({label1: predictions})

    printResults(predictions, y_test, "Для linear_model.LassoLars:")

def printResults(predictions, y_test, string, i = None):
    print(string)

    if i is not None:
        print("Для n_neighbours = ", i)
        print('mean_squared_log_error:\t%.5f' % mean_squared_log_error(y_test, predictions))

    print('mean_absolute_error:\t(пункты)%.4f' % mean_absolute_error(y_test, predictions))

    print("Median Absolute Error: " + str(round(median_absolute_error(predictions, y_test), 2)))

    RMSE = round(sqrt(mean_squared_error(predictions, y_test)), 2)

    print("Root mean_squared_error: " + str(RMSE))

# advanced regressors:

def evaluate(model, X, y, title):
    predictions = model.predict(X)
    # errors = abs(np.expm1(predictions) - np.expm1(y))
    # mape = 100 * np.mean(errors / np.expm1(y))
    # accuracy = 100 - mape
    # score_gbr = model.score(X, y)

    print(title)

    print('mean_absolute_error:\t(пункты)%.4f' % mean_absolute_error(y, predictions))
    print("Median Absolute Error: " + str(round(median_absolute_error(predictions, y), 2)))

    RMSE = round(sqrt(mean_squared_error(predictions, y)), 2)

    print("Root mean_squared_error: " + str(RMSE))
    print("\n")

    return predictions

def scatter_plot(prediction, y, title):
    plt.rcParams['figure.figsize'] = (10, 4)
    plt.style.use(style = 'ggplot')

    plt.scatter(x = prediction, y = y, alpha = .75)

    plt.ylabel('log(input price)', fontsize = 16)
    plt.xlabel('log(predicted price)', fontsize = 16)

    plt.tick_params(labelsize = 12)
    plt.title(title, fontsize = 16)

    plt.show()

def feature_extraction(importances, title):
    plt.rcParams['figure.figsize'] = (12, 6)
    importances[0:15].iloc[::-1].plot(kind = 'barh',legend = False,fontsize = 16)

    plt.tick_params(labelsize = 18)

    plt.ylabel("Feature",fontsize = 20)
    plt.xlabel("Importance viariable",fontsize = 20)
    plt.title(title,fontsize = 20)

    plt.show()

def scatter_plot2(prediction1, y1, prediction2, y2, title):
    a = min(min(prediction1),
            min(y1),
            min(prediction2),
            min(y2))-0.2
    b = max(max(prediction1),
            max(y1),
            max(prediction2),
            max(y2))+0.2

    plt.rcParams['figure.figsize'] = (10, 4)
    plt.style.use(style = 'ggplot')

    plt.scatter(x = prediction1,
                y = prediction1-y1,
                color = 'red',
                label = 'Training data',
                alpha = .75)

    plt.scatter(x = prediction2,
                y = prediction2-y2,
                color = 'blue',
                marker = 's',
                label = 'Test data',
                alpha = .75)

    plt.hlines(y = 0, xmin = a, xmax = b, color = "black")

    plt.ylabel('log(input price)',fontsize = 16)
    plt.xlabel('log(predicted price)',fontsize = 16)

    plt.tick_params(labelsize = 16)
    plt.title(title,fontsize = 16)
    plt.legend(fontsize = 16)

    plt.show()


def scatter_plot3(prediction1, y1, prediction2, y2, title):
    a = min(min(prediction1),
            min(y1),min(prediction2),
            min(y2))-0.2
    b = max(max(prediction1),
            max(y1),max(prediction2),
            max(y2))+0.2

    plt.rcParams['figure.figsize'] = (10, 4)
    plt.style.use(style = 'ggplot')

    plt.scatter(x = prediction1,
                y = y1,
                color = 'red',
                label = 'Training data',
                alpha = .75)
    plt.scatter(x = prediction2,
                y = y2,
                color = 'blue',
                marker = 's',
                label = 'Test data',
                alpha = .75)

    plt.plot([a, b], [a, b], c = "black")

    plt.ylabel('log(input price)', fontsize = 16)
    plt.xlabel('log(predicted price)', fontsize = 16)
    plt.tick_params(labelsize = 16)
    plt.title(title,fontsize = 16)
    plt.legend(fontsize = 16)

    plt.show()

def gradientBoostingRegression(x_train, x_test, y_train, y_test):
    gbr = GradientBoostingRegressor(min_samples_split = 400,
                                    min_samples_leaf = 50,
                                    subsample = 0.8,
                                    random_state = 1,
                                    learning_rate = 0.01,
                                    max_features = 'sqrt')

    param_grid = dict(n_estimators = [6000, 7000],
                      max_depth = [8, 12, 16])

    grid_gbr = GridSearchCV(gbr,
                            param_grid,
                            cv = 10,
                            scoring = 'neg_mean_squared_error',
                            n_jobs = -2)

    grid_gbr.fit(x_train, y_train)
    model_gbr = grid_gbr.best_estimator_

    printResultsAdvanced('Gradient Boosting Regression:',
                         model_gbr,
                         x_train, x_test, y_train, y_test,
                         'Gradient Boosting Regression: Training set feature importance',
                         True)

def XGBRegression(x_train, x_test, y_train, y_test):
    xgbr = xgb.XGBRegressor(random_state = 1,
                            n_jobs = -2,
                            learning_rate = 0.01)

    param_grid = dict(gamma = [0.03, 0.04],
                      max_depth = [4, 8, 12],
                      n_estimators = [3000, 4000])

    grid_xgbr = GridSearchCV(xgbr,
                             param_grid,
                             cv = 10,
                             scoring = 'neg_mean_squared_error')

    grid_xgbr.fit(x_train, y_train)

    model_xgbr = grid_xgbr.best_estimator_

    printResultsAdvanced('XGB Regression:',
                         model_xgbr,
                         x_train, x_test, y_train, y_test,
                         'XGB Regression: Training set feature importance',
                         True)

# у меня здесь убунта 18.04 зависает через 10-30 секунд после запуска обучения,
# поэтому метод закомментирован
# def randomForestRegression(x_train, x_test, y_train, y_test):
#     rf = RandomForestRegressor()
#
#     param_grid = dict(n_estimators = [3000, 4000, 5000],
#                       max_depth = [None, 4],
#                       min_samples_leaf = [1, 2])
#
#     grid_rf = GridSearchCV(rf,
#                            param_grid,
#                            cv = 10,
#                            scoring = 'neg_mean_squared_error')
#
#     grid_rf.fit(x_train, y_train)
#     model_rf = grid_rf.best_estimator_
#
#     printResultsAdvanced('Random Forest Regression:',
#                          model_rf,
#                          x_train, x_test, y_train, y_test,
#                          'Random Forest Regression: Training set feature importance',
#                          True)

def LassoRegressionAdvanced(x_train, x_test, y_train, y_test):
    lasso = Lasso(max_iter = 10000)
    param_grid = dict(alpha = np.logspace(-4, 1, 50))

    grid_lasso = GridSearchCV(lasso,
                              param_grid,
                              cv = 10,
                              scoring = 'neg_mean_squared_error',
                              n_jobs = -1)

    grid_lasso.fit(x_train, y_train)

    model_lasso = grid_lasso.best_estimator_

    printResultsAdvanced('Lasso Regression:',
                         model_lasso,
                         x_train, x_test, y_train, y_test, '')

def linearRegressionAdvanced(x_train, x_test, y_train, y_test):
    lr = LinearRegression(n_jobs = -1)
    param_grid = dict(fit_intercept = [True, False],
                      normalize = [True, False],
                      copy_X = [True, False])

    grid_lr = GridSearchCV(lr,
                           param_grid,
                           cv = 10,
                           scoring = 'neg_mean_squared_error')

    grid_lr.fit(x_train, y_train)

    model_lr = grid_lr.best_estimator_

    printResultsAdvanced('Linear Regression:',
                         model_lr,
                         x_train, x_test, y_train, y_test, '')


def printResultsAdvanced(title, model, x_train, x_test, y_train, y_test, title1, superAdvanced = False):
    title0 = title
    model_tmp = model

    title = title0 + ' training set model performance'
    prediction_train = evaluate(model_tmp, x_train, y_train, title)
    title = title0 + ' test set model performance'
    prediction_test = evaluate(model_tmp, x_test, y_test, title)

    title = title0 + ' residual plot'
    scatter_plot2(prediction_train, y_train, prediction_test, y_test, title)

    title = title0 + ' performance evaluation'
    scatter_plot3(prediction_train, y_train, prediction_test, y_test, title)

    if superAdvanced:
        importances_train = pd.DataFrame({'Feature': x_train.columns, 'Importance': model.feature_importances_})
        importances_train = importances_train.sort_values('Importance', ascending = False).set_index('Feature')
        feature_extraction(importances_train, title1)




print("\n")
print("Предсказание по цене:")
print("\n")

regressionWorker('price', 'reviews_per_month')

print("\n")
print("Предсказание по отзывам в месяц:")
print("\n")

regressionWorker('reviews_per_month', 'price')