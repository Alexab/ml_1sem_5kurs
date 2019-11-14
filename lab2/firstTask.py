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
	X = data[["minimum_nights", "number_of_reviews", "availability_365"]]
	Y = data["price"]

	h = .02

	knn=neighbors.KNeighborsClassifier()

	knn.fit(X, Y)
	
	x1_min, x1_max = X[:,0].min() - .5, X[:,0].max() + .5
	x2_min, x2_max = X[:,1].min() - .5, X[:,1].max() + .5
	
	x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

	Z = knn.predict(np.c_[x1.ravel(), x2.ravel()])

	Z = Z.reshape(x1.shape)

	pl.figure(1, figsize=(12, 9))
	pl.set_cmap(pl.cm.Paired)
	pl.pcolormesh(x1, x2, Z)

	pl.scatter(X[:,0], X[:,1],c=Y, edgecolors="b")
	
	pl.xlabel('Sepal length')
	pl.ylabel('Sepal width')
	
	pl.xlim(x1.min(), x1.max())
	pl.ylim(x2.min(), x2.max())
	pl.xticks(())
	pl.yticks(())
	
	pl.show()


def get_grid(data):
	x_min, x_max = data.iloc[:, 0].min() - 1, data.iloc[:, 0].max() + 1
	y_min, y_max = data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1
	return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

def decisionTree():
	data.info()

	crit = 'entropy'
	clf_tree = DecisionTreeClassifier(criterion = crit, max_depth = None, random_state = 20, presort = True)

	train_data = data[["minimum_nights", "number_of_reviews", "availability_365"]]
	train_labels = data["price"]

	clf_tree.fit(X = train_data, y = train_labels)
	print("\n\nГлубина дерева: {}".format(clf_tree.get_depth()))
	print("\n")

	dotfile = export_graphviz(clf_tree, feature_names=['minimum_nights', 'number_of_reviews', 'availability_365'], class_names=["N", "Y", "C"], out_file=None, filled=True, node_ids=True)
	graph = Source(dotfile)

	graph.format = 'png'
	graph.render("tree_example_tree_{}".format(crit),view = True)

	xx, yy = get_grid(train_data)
	predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, predicted, cmap='autumn')
	plt.scatter(train_data.iloc[:, 0], train_data.iloc[:, 1], c=train_labels, s=100, cmap='autumn', edgecolors='black', linewidth=1.5)
	plt.xlabel("price")
	plt.ylabel("ticket_class")
	plt.savefig("tree_example_surf_{}".format(crit))
	plt.show()
	

def naiveBayes():
	X = data[["minimum_nights", "number_of_reviews", "availability_365"]]
	Y = data["price"]

	model = GaussianNB()

	model.fit(X, Y)

	predicted = model.predict([[1,2],[3,4]])
	print(predicted)


def show_2d(x_hor, x_ver, y, x_hor_name='', x_ver_name='', y_names=None, subplot=None, type2=False):
	"""
	Функция вывода диаграммы с точками в двумерном пространстве
	:param x1: признак, будет по оси Х
	:param x2: признак, будет по оси У
	:param y: класс, будет показан цветом
	:return: None
	"""
	sub = plt.gca() if subplot is None else subplot
	edgecolors = "r" if type2 else "b"
	# Если не заданы имена классов, выведем их номера
	if y_names is None:
		y_names = y_names.astype(np.str).tolist()
	# Тупой вывод диаграммы с точками разных цветов (классов). Цикл по классам.
	for yi in np.unique(y):
		i = np.where(y == yi)
		sub.scatter(x_hor[i], x_ver[i], label=y_names[yi], edgecolors=edgecolors)
	# Подпишем оси
	sub.set_xlabel(x_hor_name)
	sub.set_ylabel(x_ver_name)
	# Укажем легенду, если это не подграфик
	if subplot is None:
		sub.legend()

def supportVectorMachine():
	# ------------------------------------------------------------------------------------------------------------------- #
	# Исходные параметры
	# ------------------------------------------------------------------------------------------------------------------- #
	# Наша игрушечная база в формате pandas DataFrame
	data = load_iris()
	labels = data.target
	samples = data.data
	print("Dataset Iris:\n\tx_data shape: {}\n\tfeatures: {}".format(samples.shape, data.feature_names))

	# ------------------------------------------------------------------------------------------------------------------- #
	# 1. Посмотрим на графики признаки-класс
	# ------------------------------------------------------------------------------------------------------------------- #

	x0, x0_name = samples[:,0], data.feature_names[0]
	x1, x1_name = samples[:,1], data.feature_names[1]
	x2, x2_name = samples[:,2], data.feature_names[2]
	x3, x3_name = samples[:,3], data.feature_names[3]
	y, y_names  = labels, data.target_names

	# Выведем сразу все возможные двумерные графики
	fig, axs = plt.subplots(2, 3)
	show_2d(x_hor=x0, x_ver=x1, y=y, x_hor_name=x0_name, x_ver_name=x1_name, y_names=y_names, subplot=axs[0, 0])
	show_2d(x_hor=x0, x_ver=x2, y=y, x_hor_name=x0_name, x_ver_name=x2_name, y_names=y_names, subplot=axs[0, 1])
	show_2d(x_hor=x0, x_ver=x3, y=y, x_hor_name=x0_name, x_ver_name=x3_name, y_names=y_names, subplot=axs[0, 2])
	show_2d(x_hor=x1, x_ver=x2, y=y, x_hor_name=x1_name, x_ver_name=x2_name, y_names=y_names, subplot=axs[1, 0])
	show_2d(x_hor=x1, x_ver=x3, y=y, x_hor_name=x1_name, x_ver_name=x3_name, y_names=y_names, subplot=axs[1, 1])
	show_2d(x_hor=x2, x_ver=x3, y=y, x_hor_name=x2_name, x_ver_name=x3_name, y_names=y_names, subplot=axs[1, 2])
	# Легенда нужна одна на все рафики
	handles, labels = axs[0,0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper center')
	# Выведем, что получилось
	#fig.show()

	# ------------------------------------------------------------------------------------------------------------------- #
	# 2. Выберем два первых признака и приравняем третий класс к первому
	# ------------------------------------------------------------------------------------------------------------------- #

	y2 = np.copy(y)
	y2[y2 == 2] = 1
	y2_names = np.copy(y_names[:2])
	plt.figure()
	show_2d(x_hor=x0, x_ver=x1, y=y2, x_hor_name=x0_name, x_ver_name=x1_name, y_names=y2_names)
	#plt.show()

	# ------------------------------------------------------------------------------------------------------------------- #
	# 3. Разделим базу на обучающую и тестовую выборку
	# ------------------------------------------------------------------------------------------------------------------- #

	plt.figure()
	data_train, data_test, labels_train, labels_test = train_test_split(samples[:, :2], y2, test_size = 0.1, random_state=20)
	show_2d(x_hor=data_train[:, 0], x_ver=data_train[:, 1], y=labels_train, x_hor_name=x0_name, x_ver_name=x1_name, y_names=y2_names)
	show_2d(x_hor=data_test[:, 0],  x_ver=data_test[:, 1],  y=labels_test,  x_hor_name=x0_name, x_ver_name=x1_name, y_names=y2_names, type2=True)
	#plt.show()

	# ------------------------------------------------------------------------------------------------------------------- #
	# 4. Выполним обучение по соответствующей выборке
	# ------------------------------------------------------------------------------------------------------------------- #

	# Приведем обучающие примеры к нужному формату
	n = data_train.shape[0]
	dtrain1 = np.expand_dims(data_train[:, 0], axis = -1)
	dtrain2 = np.expand_dims(data_train[:, 1], axis = -1)
	dtrainy = np.expand_dims(labels_train, axis=-1)
	dtrainy[dtrainy == 0] = -1

	w1 = np.zeros((n, 1))
	w2 = np.zeros((n, 1))

	epochs = 1
	alpha = 0.0001 # lr

	while (epochs < 1000):
		y = w1 * dtrain1 + w2 * dtrain2
		prod = y * dtrainy
		count = 0
		for val in prod:
			if (val >= 1):
				cost = 0
			w1 = w1 - alpha * (2 * 1 / epochs * w1)
			w2 = w2 - alpha * (2 * 1 / epochs * w2)

		else:
			cost = 1 - val
			w1 = w1 + alpha * (dtrain1[count] * dtrainy[count] - 2 * 1 / epochs * w1)
			w2 = w2 + alpha * (dtrain2[count] * dtrainy[count] - 2 * 1 / epochs * w2)
			count += 1
			epochs += 1

		if epochs % 100 == 0:
			print("epoch: {}".format(epochs))


	print("fit done!")

	# ------------------------------------------------------------------------------------------------------------------- #
	# 5. Выполним тест по соответствующей выборке
	# ------------------------------------------------------------------------------------------------------------------- #

	# Приведем тестовые примеры к нужному формату
	n = data_test.shape[0]
	dtest1 = np.expand_dims(data_test[:, 0], axis = -1)
	dtest2 = np.expand_dims(data_test[:, 1], axis = -1)
	dtesty = np.expand_dims(labels_test, axis=-1)
	dtesty[dtesty == 0] = -1
	w1 = w1[:n]
	w2 = w2[:n]

	# Предсказываем
	y_pred = w1 * dtest1 + w2 * dtest2
	predictions = []
	for val in y_pred:
		if(val > 1):
			predictions.append(1)
		else:
			predictions.append(-1)

	print("Accuracy: {}".format(accuracy_score(dtesty, predictions)), end="\n\n")

	# ------------------------------------------------------------------------------------------------------------------- #
	# 6. Более продвинутый вариант
	# ------------------------------------------------------------------------------------------------------------------- #

	# Приведем обучающие примеры к нужному формату
	n = data_train.shape[0]
	dtrain1 = np.expand_dims(data_train[:, 0], axis = -1)
	dtrain2 = np.expand_dims(data_train[:, 1], axis = -1)
	dtrainy = np.expand_dims(labels_train, axis=-1)
	dtrainy[dtrainy == 0] = -1
	w1 = np.zeros((n, 1))
	w2 = np.zeros((n, 1))

	n = data_test.shape[0]
	dtest1 = np.expand_dims(data_test[:, 0], axis = -1)
	dtest2 = np.expand_dims(data_test[:, 1], axis = -1)
	dtesty = np.expand_dims(labels_test, axis=-1)
	dtesty[dtesty == 0] = -1

	epochs = 1
	alpha = 0.0001 # lr

	while (epochs < 500):
		y = w1 * dtrain1 + w2 * dtrain2
		prod = y * dtrainy
		count = 0
		for val in prod:
			if (val >= 1):
				cost = 0
				w1 = w1 - alpha * (2 * 1 / epochs * w1)
				w2 = w2 - alpha * (2 * 1 / epochs * w2)
			else:
				cost = 1 - val
				w1 = w1 + alpha * (dtrain1[count] * dtrainy[count] - 2 * 1 / epochs * w1)
				w2 = w2 + alpha * (dtrain2[count] * dtrainy[count] - 2 * 1 / epochs * w2)
			count += 1
		epochs += 1

		# Сразу затестим
		if epochs % 10 == 0:
			w12 = np.sort(w1, axis=0)[:n]
			w22 = np.sort(w2, axis=0)[:n]
			# Предсказываем
			y_pred = w12 * dtest1 + w22 * dtest2
			predictions = []
			for val in y_pred:
				if (val > 1):
					predictions.append(1)
				else:
					predictions.append(-1)

			print("epoch: {}\tacc: {}".format(epochs, accuracy_score(dtesty, predictions)))


	print("fit done!", end="\n\n")

	w1_n = w1[np.nonzero(w1)]
	w2_n = w2[np.nonzero(w2)]

	# ------------------------------------------------------------------------------------------------------------------- #
	# 7. Вариант из sklearn
	# ------------------------------------------------------------------------------------------------------------------- #

	clf = SVC(kernel='linear')
	clf.fit(data_train,labels_train)
	y_pred = clf.predict(data_test)
	print("sklearn accuracy: {}".format(accuracy_score(labels_test,y_pred)))
	

#1
#kNN()
#2
#decisionTree()
#3
#naiveBayes()
#4
#supportVectorMachine()
	


