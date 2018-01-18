#%%
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=plt.cm.binary, interpolation="nearest")
plt.axis('off')
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# shuffle training set
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


#%%
from sklearn.linear_model import SGDClassifier

# train a binary "5" classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

#%%
# evaluate the model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)

#%%
# play with precision and recall
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#%%
# roc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewith=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


plot_roc_curve(fpr, tpr)
plt.show()

roc_auc_score(y_train_5, y_scores)

#%%
# try random forest classifier and compare roc with sgd
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method='predict_proba')

y_scores_forest = y_probas_forest[:, 1]  #  score= proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(
    y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend("lower right")

roc_auc_score(y_train_5, y_scores_forest)

#%% multiclass classifier
from sklearn.preprocessing import StandardScaler

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# scale features and re-evaluate
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

#%%
# error analysis
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

#%%
# multilabel classification
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1_score(y_train, y_train_knn_pred, average='macro')

#%%
#  multioutput classification
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[3450]])


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


plot_digit(clean_digit)

#%%
# Exercise 1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

weights = ['uniform', 'distance']  
numNeighbors = [3, 4, 5]
param_grid = dict(weights=weights, n_neighbors=numNeighbors)


grid = RandomizedSearchCV(KNeighborsClassifier(), param_grid, cv=3, n_iter=5)
grid.fit(X_train, y_train)

cross_val_score(grid, X_test, y_test, cv=3, scoring="accuracy")

# %%
#  Exercise 2
from scipy.ndimage.interpolation import shift
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

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
shuffle_index = np.random.permutation(60000)
X_train_augumented, y_train_augumented = X_train_augumented[shuffle_index], y_train_augumented[shuffle_index]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_augumented, y_train_augumented)

cross_val_score(knn_clf, X_test, y_test, cv=3, scoring="accuracy")