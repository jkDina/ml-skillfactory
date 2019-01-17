import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)
data = pd.read_csv('hw_data.csv')
a=data.head()
#print(a)

labels = data['quality']
del(data['quality'])

data_train, data_test, labels_train, labels_test = train_test_split(data, labels)

#model = LinearRegression()
#model = DecisionTreeRegressor(max_depth=12)
model = RandomForestRegressor(max_depth=21, n_estimators=12)
model.fit(data_train, labels_train)

prediction = model.predict(data_test)

ms_err = mean_squared_error(prediction, labels_test)
print(ms_err)

data_control = pd.read_csv('data_to_estimate.csv')
prediction = model.predict(data_control)

print(",".join(["{:.3f}".format(num) for num in prediction]))





