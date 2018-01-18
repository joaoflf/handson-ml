from sklearn.datasets import fetch_mldata
import numpy as np

mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# shuffle training set
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

from scipy.ndimage.interpolation import shift

X_train_augumented = X_train.tolist()
y_train_augumented = y_train.tolist()



for i in range(X_train.shape[0]):
    image = X_train[i].reshape(28, 28)
    shift_up = shift(image, [-1, 0],  cval=0).reshape([-1])
    shift_down = shift(image, [1, 0],  cval=0).reshape([-1])
    shift_right = shift(image, [0, 1],  cval=0).reshape([-1])
    shift_left = shift(image, [0, -1],  cval=0).reshape([-1])

    X_train_augumented.append(shift_up)
    X_train_augumented.append(shift_down)
    X_train_augumented.append(shift_right)
    X_train_augumented.append(shift_left)

    y_train_augumented.append(y_train[i])
    y_train_augumented.append(y_train[i])
    y_train_augumented.append(y_train[i])
    y_train_augumented.append(y_train[i])


X_train_augumented = np.array(X_train_augumented)
y_train_augumented = np.array(y_train_augumented)
print(X_train_augumented.shape)
print(y_train_augumented.shape)