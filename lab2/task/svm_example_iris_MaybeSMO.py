import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

# Укажем цветовую схему (набор цветов), которую хотим использовать по дефолту
# plt.set_cmap("Set1")

def show_iris_2d(x_hor, x_ver, y, x_hor_name='', x_ver_name='', y_names=None, subplot=None, type2=False):
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
show_iris_2d(x_hor=x0, x_ver=x1, y=y, x_hor_name=x0_name, x_ver_name=x1_name, y_names=y_names, subplot=axs[0, 0])
show_iris_2d(x_hor=x0, x_ver=x2, y=y, x_hor_name=x0_name, x_ver_name=x2_name, y_names=y_names, subplot=axs[0, 1])
show_iris_2d(x_hor=x0, x_ver=x3, y=y, x_hor_name=x0_name, x_ver_name=x3_name, y_names=y_names, subplot=axs[0, 2])
show_iris_2d(x_hor=x1, x_ver=x2, y=y, x_hor_name=x1_name, x_ver_name=x2_name, y_names=y_names, subplot=axs[1, 0])
show_iris_2d(x_hor=x1, x_ver=x3, y=y, x_hor_name=x1_name, x_ver_name=x3_name, y_names=y_names, subplot=axs[1, 1])
show_iris_2d(x_hor=x2, x_ver=x3, y=y, x_hor_name=x2_name, x_ver_name=x3_name, y_names=y_names, subplot=axs[1, 2])
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
show_iris_2d(x_hor=x0, x_ver=x1, y=y2, x_hor_name=x0_name, x_ver_name=x1_name, y_names=y2_names)
#plt.show()

# ------------------------------------------------------------------------------------------------------------------- #
# 3. Разделим базу на обучающую и тестовую выборку
# ------------------------------------------------------------------------------------------------------------------- #

plt.figure()
data_train, data_test, labels_train, labels_test = train_test_split(samples[:, :2], y2, test_size = 0.1, random_state=20)
show_iris_2d(x_hor=data_train[:, 0], x_ver=data_train[:, 1], y=labels_train, x_hor_name=x0_name, x_ver_name=x1_name,
             y_names=y2_names)
show_iris_2d(x_hor=data_test[:, 0],  x_ver=data_test[:, 1],  y=labels_test,  x_hor_name=x0_name, x_ver_name=x1_name,
             y_names=y2_names, type2=True)
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


# https://github.com/Madhu009/Deep-math-machine-learning.ai/blob/master/SVM-FromScratch.ipynb
