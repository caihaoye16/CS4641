import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
import os, cv2, random

import pandas as pd
import csv as csv

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    f.write('%d\n\n' %test_scores_mean[-1])
    # f.write('\n')
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# train_df = pd.read_csv('train.csv', header=0) 

# label = shuffle(train_df["label"].values, n_samples = 10000, random_state = 0)

# y_label = []
# for i in label:
#     temp = [0,0,0,0,0,0,0,0,0,0]
#     temp[i] = 1
#     y_label.append(temp)

# X = shuffle(train_df.drop(["label"], axis = 1).values, n_samples = 10000, random_state = 0)
# print(X.shape)
# print(label.shape)

TRAIN_DIR = 'train/'
# TEST_DIR = '../input/test/'

ROWS = 128
COLS = 128
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for ful

def read_image(file_path):
	img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
	img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
	# cv2.imwrite('./preprocessed/' + file_path, img)
	return img

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    return data
train = prep_data(train_images).astype(float)

# trx = []
for i, im in enumerate(train):
    # im = im.reshape((28,28))
    # trx.append(hog(im))
    train[i]=hog(im)

labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)
labels = np.array(labels)
print(labels)



title = "Learning Curves (kNN)"
k_val = [2,3,4,5,6,7,8,9,10]
# cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
f = open('knn_acc.txt', 'w')
for i in k_val:
    estimator = KNeighborsClassifier(n_neighbors=i)
    print(i)
    f.write('%d\n' %i)
    plot_learning_curve(estimator, title, train, labels, (0.5, 1.01), cv=5, n_jobs=4)
    
    # f.write('\n')
    plt.savefig('LC_knn_kval_' + str(i) + '.png')
    # joblib.dump(estimator, 'knn' + str(i) + '.pkl')
f.close()
