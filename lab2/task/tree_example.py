# pip3 install [--user] graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import export_graphviz, plot_tree
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier

# Наша игрушечная база в формате pandas DataFrame
data = np.array([[3, 2, 1], [12, 3, 1], [27, 1, 1], [14, 2, 1],
                 [35, 1, 1], [35, 3, 0], [19, 1, 0], [65, 1, 0],
                 [29, 2, 1], [31, 3, 1], [49, 3, 0], [65, 1, 1],
                 [71, 2, 0], [20, 3, 0]])
data = pd.DataFrame(data=data, columns=["Age", "Class", "Survived"])

# Так как записей всего 14, выведем их все
print(data.head(n=14))

# -------------------------------------------------------------
# построим дерево
# -------------------------------------------------------------

# Укажем критерий качества разбиения
crit = 'entropy'
# Создаем объект-классификатор с заданными параметрами
clf_tree = DecisionTreeClassifier(criterion=crit, max_depth=None, random_state=20, presort=True)

# Разделим общую таблицу данных на признаки и метки
train_data = data[["Age", "Class"]]
train_labels = data["Survived"]

# обучим дерево и выведем его глубину
clf_tree.fit(X=train_data, y=train_labels)
print("Глубина дерева: {}".format(clf_tree.get_depth()))

# Посмотрим само дерево
# plot_tree(clf_tree, feature_names=['age', 'ticket_class'], class_names=["Y", "N"], node_ids=True, impurity=True) # для тех, у кого graphviz не заработал, ущербный вариант, но хоть что-то
dotfile = export_graphviz(clf_tree, feature_names=['age', 'class'], class_names=["N", "Y"], out_file=None, filled=True, node_ids=True)
graph = Source(dotfile)
# Сохраним дерево как toy_example_tree_X.png, где Х - entropy или gini, критерий качестве рабиения
graph.format = 'png'
graph.render("tree_example_tree_{}".format(crit),view=True)

# Отобразим плоскость с разделением классов - так, как этому обучилось дерево
# Вспомогательная функция, которая будет возвращать сетку значений-координат для дальнейшей визуализации.
def get_grid(data):
    x_min, x_max = data.iloc[:, 0].min() - 1, data.iloc[:, 0].max() + 1
    y_min, y_max = data.iloc[:, 1].min() - 1, data.iloc[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xx, yy = get_grid(train_data)
predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, predicted, cmap='autumn')
plt.scatter(train_data.iloc[:, 0], train_data.iloc[:, 1], c=train_labels, s=100, cmap='autumn', edgecolors='black', linewidth=1.5)
plt.xlabel("age")
plt.ylabel("ticket_class")
plt.savefig("tree_example_surf_{}".format(crit))
plt.show()

