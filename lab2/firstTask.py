# created by БорискинМА
# 01.11.19,10.11.19,14.11.19
# gedit
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import graphviz
import folium
import random
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')
import sys
import csv
import math
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from sklearn import neighbors, datasets 
from matplotlib import pyplot as plt
from matplotlib import style
from folium import plugins
from folium.plugins import HeatMap
from itertools import islice
from sklearn.tree import export_graphviz, plot_tree
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = pd.read_csv('AB_NYC_2019.csv')

def totalABNYC2019():
	print('Количество записей в базе:', len(set(data['id'])), '\n')

def histogram():

	data.info()
	print('\n')	

	data.hist(figsize=(15,8), layout=(2,5))

	plt.show()

def areThereAnyMissingItems():
	print(data.isnull().sum())
	print('\n')
	print ('Broken items in', 'столбцы \"дата последнего отзыва\" и \"количество отзывов в месяц\"', 'are the same?', (data['last_review'].isnull() == data['reviews_per_month'].isnull()).all())
	print('\n')

def fillMissingItems():
	data.loc[data['reviews_per_month'].isnull(), 'reviews_per_month'] = 0
	
def checkNumericVariables():
	print(data.describe())
	print('\n')

def removeUnnecessaryColumns():
	data.drop(['id','name','host_name','last_review'], axis = 1, inplace = True)

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
	data_loc = data[['latitude','longitude']].values
	data_loc = data_loc.tolist()
	HeatMap(data[['latitude','longitude']].dropna(), radius = 8, gradient = {0.2:'blue', 0.4:'purple', 0.6:'orange', 1.0:'red'}).add_to(m)

	#if you want you can open it manually in your webbrowser
	m.save("heatMap.html")

def correlation():
	print(set(data['neighbourhood_group']))
	print('\n')

	for group in set(data['neighbourhood_group']):
		print(group)
		print(data[data['neighbourhood_group'] == group][['price','minimum_nights', 'availability_365']].corr())
		print('\n')
	sns.distplot(data['minimum_nights'])
	plt.show()

	corr = data[data['neighbourhood_group'] == group][['price','minimum_nights', 'availability_365']].corr()
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


#1
totalABNYC2019()
#2
histogram()
#3
areThereAnyMissingItems()
fillMissingItems()
checkNumericVariables()
removeUnnecessaryColumns()
data = cleanAbnormalities()
#4
latlonNbrhdGr()
heatMap()
#5
correlation()
#6
latlonPrice()
#7
wordsCounter()


#def kNN():


def get_grid(data):
	x_min, x_max = data.iloc[:, 0].min() - 1, data.iloc[:, 0].max() + 1
	y_min, y_max = data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1
	return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))	

def decisionTree():
	# -------------------------------------------------------------
	# построим дерево
	# -------------------------------------------------------------

	# Укажем критерий качества разбиения
	crit = 'entropy'
	# Создаем объект-классификатор с заданными параметрами
	clf_tree = DecisionTreeClassifier(criterion=crit, max_depth=None, random_state=20, presort=True)

	#'latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365','price'

	# Разделим общую таблицу данных на признаки и метки
	train_data = data[['latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365']]
	train_labels = data['price']

	# обучим дерево и выведем его глубину
	clf_tree.fit(X=train_data, y=train_labels)
	print("Глубина дерева: {}".format(clf_tree.get_depth()))

	# Посмотрим само дерево
	# plot_tree(clf_tree, feature_names=['age', 'ticket_class'], class_names=["Y", "N"], node_ids=True, impurity=True) # для тех, у кого graphviz не заработал, ущербный вариант, но хоть что-то
	dotfile = export_graphviz(clf_tree, feature_names=['latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365'], class_names=True, out_file=None, filled=True, node_ids=True)
	graph = Source(dotfile)
	# Сохраним дерево как toy_example_tree_X.png, где Х - entropy или gini, критерий качестве рабиения
	graph.format = 'png'
	graph.render("tree_example_tree_{}".format(crit),view=True)

	# Отобразим плоскость с разделением классов - так, как этому обучилось дерево
	# Вспомогательная функция, которая будет возвращать сетку значений-координат для дальнейшей визуализации.
	
	xx, yy = get_grid(train_data)
	predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, predicted, cmap='autumn')
	plt.scatter(train_data.iloc[:, 0], train_data.iloc[:, 1], c=train_labels, s=100, cmap='autumn', edgecolors='black', linewidth=1.5)
	plt.xlabel("age")
	plt.ylabel("ticket_class")
	plt.savefig("tree_example_surf_{}".format(crit))
	plt.show()




#def naiveBayes():


#def supportVectorMachine():
	
	

#1
#kNN()
#2
decisionTree()
#3
#naiveBayes()
#4
#supportVectorMachine()
	


