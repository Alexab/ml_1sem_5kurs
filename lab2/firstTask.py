# created by БорискинМА
# 01.11.19,10.11.19
# gedit
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
import graphviz
import folium
import random
import matplotlib.cm as cm
from sklearn import neighbors, datasets 
from matplotlib import pyplot as plt
from matplotlib import style
from folium import plugins
from folium.plugins import HeatMap
from itertools import islice

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
