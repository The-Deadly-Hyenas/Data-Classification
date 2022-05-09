"""
Starting script to try out different classifiers.

Trying out different classifiers and grid searching hyperparameters
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
import json
import time

# get start time
t0 = time.time()

# names of the used classifiers
names = [
    "Nearest Neighbors",
    "SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# grid search parameters for the different classifiers
parameters = [
    # knn
    {
        'n_neighbors': [3, 15, 25],
        'weights': ['uniform', 'distance'],
        'leaf_size': [10, 15, 20, 30]
    },
    # svm
    {
        'C': [0.5, 1],
        'kernel': ['poly', 'rbf'],
        'degree': [3, 5, 6],
        'gamma': ['scale', 'auto']
    },
    # Decision Tree
    {
        'max_depth': [3, 5, 7, 15],
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    # RandomForestClassifier
    {
        'bootstrap': [True, False],
        'max_depth': [90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 800]
    },
    # MLPClassifier
    {
        'hidden_layer_sizes': [(10, 30, 10), (20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    },
    #  AdaBoostClassifier
    {
        'base_estimator__max_depth': [i for i in range(2, 11, 2)],
        'base_estimator__min_samples_leaf': [5, 10],
        'n_estimators': [10, 50, 250, 1000],
        'learning_rate': [0.01, 0.1]
    },
    # GaussianNB
    {
        'priors': [None],
        'var_smoothing': [1e-9]
    },
    # QuadraticDiscriminantAnalysis
    {
        'priors': [None],
        'reg_param': [0.0],
    }

]

# load the data and reset index of dataframe
df: pd.DataFrame = pd.read_pickle(
    "../training_dataset_task3/task_3_training_5bdf9a9ed30b9a66_749fa46_pandas.pkl").reset_index()

# get only the low and mid level features + segment_id
X = df.loc[:, "essentia_dissonance_mean":"mirtoolbox_roughness_pct_90"]
# target value
y = df["quadrant"]

# preprocess dataset
X = StandardScaler().fit_transform(X)

# split the data according to segment_id
# store the splits as tuple (train indices, test_indices)
# for example the training indices are the first 26 segments
# and the test_indices is the last segment 27
cv = []

for i in range(27):
    train_indices = df[df["segment_id"] != i].index.to_list()
    test_indices = df[df["segment_id"] == i].index.to_list()
    cv.append((train_indices, test_indices))

# header for result output
print(f"{'Classifier' : <20} {'score' : >5}")

# iterate over classifiers
for name, clf, params in zip(names, classifiers, parameters):
    # grid search the parameters for a given classifier
    gs_cv = GridSearchCV(clf, params, cv=cv, n_jobs=6)
    gs_cv.fit(X, y)
    score = gs_cv.score(X, y)

    # save best model parameters to results file
    with open('results.txt', mode="a+") as f:
        f.write(name + "\n")
        f.write("score: " + str(score) + "\n")
        f.write(json.dumps(gs_cv.best_params_) + "\n\n")

    print(f"{name:<20} {score: >5}")
    print(gs_cv.best_params_)

# print time it took to run script
t1 = time.time()
print(t1 - t0)
