from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from plot_tree import plot_tree

data = pd.read_csv('credit_homework.csv', index_col='id')
target = data['Вернул кредит']
del(data['Вернул кредит'])


train_data, control_data, train_target, control_target =\
                        train_test_split(data, target, random_state=31337)
max_control_score = -1
max_md = 0
for md in range(1,25):
    model = DecisionTreeClassifier(max_depth=md, random_state=31337)
    model.fit(train_data, train_target)

    #plot_tree(model, feature_names=data.columns, class_names=['не вернет', 'вернет'])
    train_predictions = model.predict(train_data)
    control_predictions = model.predict(control_data)

    train_score = accuracy_score(train_target, train_predictions, normalize=True)
    control_score = accuracy_score(control_target, control_predictions, normalize=True)
    print(md)
    print("Точность работы классификатора на тренировочной выборке: {}%".format(train_score * 100))
    print("Точность работы классификатора на контрольной выборке: {}%".format(control_score * 100))
    if control_score > max_control_score:
        max_md = md
        max_control_score = control_score
print('max depth = ', max_md)
print(max_control_score)
