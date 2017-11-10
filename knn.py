import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math

df = pd.read_csv('auto-mpg.csv')
data = df.values

print(df.head())
print(df.describe())

np.random.seed(4)
print('----------')
np.random.shuffle(data)
print(data)

train = data[ : math.floor(len(data)*0.8)]
test = data[ math.floor(len(data)*0.8) : ]
x_train = train[:, 1:]
y_train = train[:, 0]
x_test = test[:, 1:]
y_test = test[:, 0]
print(len(x_train))
print(len(x_test))

def get_distances(x, point):
    diffs = np.subtract(point, x)
    diffs_sq = np.square(diffs)
    diffs_sq = np.sum(diffs_sq, axis=1)
    distance = np.sqrt(diffs_sq)
    return distance
point = np.array([[8, 307, 130, 3504, 12],[8, 350, 165, 3693, 11.5]])
x = np.array([10, 350, 150, 3600, 15])
print(get_distances(x, point))

def find_neighbors(dists, y, k):
    indices = np.argsort(dists)[ : k]
    return np.take(y, indices)

def avg_neighbors(neighbors):
    return np.mean(np.array(neighbors))
print(avg_neighbors([ 107.13542831, 94.28812226]))

def knn_predict(x_train, y_train, x_test, k):
    predict = []
    for x in x_test:
        distance = get_distances(x_train, x)
        neighbors = find_neighbors(distance, y_train, k)
        mpg = avg_neighbors(neighbors)
        predict.append(mpg)
    return np.round(predict, decimals=0)
print(knn_predict(x_train, y_train, x_test, 25))
print(y_test)

def mse(predicted, actual):
    return np.sum(np.square(np.subtract(predicted, actual))) / len(predicted)
predicted = knn_predict(x_train, y_train, x_test, 25)
print('----- predicted data -----')
print(predicted)
print('----- actual data -----')
print(y_test)
print('----- mean square error -----')
print(mse(predicted, y_test))

def test_knn(x_train, y_train, x_test, y_test, max_neighbors):
    mse_value = []
    k = 1
    while (k <= max_neighbors):
        predicted = knn_predict(x_train, y_train, x_test, k)
        mse_value.append([k, mse(predicted, y_test)])
        k += 1
    return sorted(mse_value, key=lambda x: x[1])[1]
print(test_knn(x_train, y_train, x_test, y_test, 324))
