"""Для работы с большими данными"""
import numpy as np
import pandas as pd

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
    ax.legend(wedges, labels, title = "Neighbourhood Groups", loc = "center left", bbox_to_anchor = (1, 0, 0.5, 1))
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

# understandingData()
#
# descriptiveAnalysis()
# diagnosticAnalysis()
# predictiveAnalysis()
#
# dataVisualization()


def kNNRegression(label1, label2):
    df_knn = data[['latitude',
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

    df_norm = df_norm[(pd.notnull(data['latitude'])) &
                      (pd.notnull(data['longitude'])) &
                      (pd.notnull(data['minimum_nights'])) &
                      (pd.notnull(data['number_of_reviews'])) &
                      (pd.notnull(data[label2])) &
                      (pd.notnull(data['calculated_host_listings_count'])) &
                      (pd.notnull(data['availability_365'])) &
                      (pd.notnull(data[label1]))]

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

    x_train = x_train[(pd.notnull(data['latitude'])) &
                      (pd.notnull(data['longitude'])) &
                      (pd.notnull(data['minimum_nights'])) &
                      (pd.notnull(data['number_of_reviews'])) &
                      (pd.notnull(data[label2])) &
                      (pd.notnull(data['calculated_host_listings_count'])) &
                      (pd.notnull(data['availability_365']))]

    x_train = x_train.dropna()
    x_train = x_train.round(6)
    x_train.apply(pd.to_numeric)

    x_test = x_test[(pd.notnull(data['latitude'])) &
                    (pd.notnull(data['longitude'])) &
                    (pd.notnull(data['minimum_nights'])) &
                    (pd.notnull(data['number_of_reviews'])) &
                    (pd.notnull(data[label2])) &
                    (pd.notnull(data['calculated_host_listings_count'])) &
                    (pd.notnull(data['availability_365']))]

    x_test = x_test.dropna()
    x_test = x_test.round(6)
    x_test.apply(pd.to_numeric)

    y_train.index.name = label1
    y_test.index.name = label1

    y_train = y_train[(pd.notnull(data[label1]))]

    y_train = y_train.dropna()
    y_train = y_train.round(6)
    y_train.apply(pd.to_numeric)

    y_test = y_test[(pd.notnull(data[label1]))]

    y_test = y_test.dropna()
    y_test = y_test.round(6)
    y_test.apply(pd.to_numeric)

    n_neighbors_array = [1, 3, 5, 7, 10, 15]

    for i in n_neighbors_array:
        knn = KNeighborsRegressor(n_neighbors = i)
        knn.fit(x_train, y_train)

        predictions = knn.predict(x_test)
        predictions = pd.DataFrame({label1: predictions})

        print("Для n_neighbours = ", i)
        print('mean_absolute_error:\t(пункты)%.4f' % mean_absolute_error(y_test, predictions))

        print('mean_squared_log_error:\t%.5f' % mean_squared_log_error(y_test, predictions))

        print("Median Absolute Error: " + str(round(median_absolute_error(predictions, y_test), 2)))

        RMSE = round(sqrt(mean_squared_error(predictions, y_test)), 2)

        print("Root mean_squared_error: " + str(RMSE))
        print("\n")

#def decisionTreeRegression():





print("\n")
print("Предсказание по цене:")
print("\n")

kNNRegression('price', 'reviews_per_month')

print("\n")
print("Предсказание по отзывам в месяц:")
print("\n")

kNNRegression('reviews_per_month', 'price')