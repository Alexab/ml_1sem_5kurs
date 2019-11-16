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


def kNN():
	df_knn = data[['latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365','price']]
	df_knn.apply(pd.to_numeric)
	
	from sklearn.utils import shuffle
	df_knn = shuffle(df_knn)

	df_norm = (df_knn[['latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365']] - df_knn[['latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365']].min()) / (df_knn[['latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365']].max() - df_knn[['latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365']].min())
	df_norm = pd.concat([df_norm, df_knn[['price']]],  axis=1)

	df_norm = df_norm[(pd.notnull(data['latitude']))&(pd.notnull(data['longitude']))&(pd.notnull(data['minimum_nights']))&(pd.notnull(data['reviews_per_month']))&(pd.notnull(data['calculated_host_listings_count']))&(pd.notnull(data['number_of_reviews']))&(pd.notnull(data['availability_365']))&(pd.notnull(data['price']))]
	df_norm = df_norm.round(6)
	df_norm = df_norm.dropna()
	df_norm.apply(pd.to_numeric)

	x_train, x_test, y_train, y_test = train_test_split(df_norm[['latitude','longitude','minimum_nights','reviews_per_month','calculated_host_listings_count','number_of_reviews','availability_365']],df_norm['price'], test_size=0.2, random_state = 42)
	print(len(x_train))
	print(len(x_test))

	print(len(y_train))
	print(len(y_test))
	print("\n")

	knn = KNeighborsRegressor(n_neighbors=7)
	knn.fit(x_train, y_train)
	predictions = knn.predict(x_test)
	predictions = pd.DataFrame({'price': predictions})

	print('mean_absolute_error:\t$%.2f' % mean_absolute_error(y_test, predictions))
	print('mean_squared_log_error:\t%.5f' % mean_squared_log_error(y_test, predictions))
	print("Median Absolute Error: " + str(round(median_absolute_error(predictions, y_test), 2)))
	RMSE = round(sqrt(mean_squared_error(predictions, y_test)), 2)
	print("Root mean_squared_error: " + str(RMSE))


#def decisionTree():
	

#def naiveBayes():


#def supportVectorMachine():
	
	

#1
kNN()
#2
#decisionTree()
#3
#naiveBayes()
#4
#supportVectorMachine()
	


