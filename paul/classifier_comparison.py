"""
Starting script to try out different classifiers.

Currently, not doing some special feature selection, splitting or trying out hyperparameter
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

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# load the data
df = pd.read_pickle("../training_dataset_task3/task_3_training_5bdf9a9ed30b9a66_749fa46_pandas.pkl")

# get only the low and mid level features
X = df.loc[:, "essentia_dissonance_mean":"mirtoolbox_roughness_pct_90"]
# target value
y = df["quadrant"]

# preprocess dataset
X = StandardScaler().fit_transform(X)

# k fold for cross validation
kf = KFold(n_splits=10)

# header for result output
print(f"{'Classifier' : <20} {'score' : >5}")

# iterate over classifiers
for name, clf in zip(names, classifiers):
    # get cross_val_scores
    score = cross_val_score(estimator=clf, X=X, y=y, cv=kf)

    # print classifier and the mean cross_val_score
    print(f"{name:<20} {score.mean(): >5}")

# results:

# Classifier           score
# Nearest Neighbors    0.428187006818356
# Linear SVM           0.43955795602543474
# RBF SVM              0.31957021374396694
# Decision Tree        0.4014211292423198
# Random Forest        0.40320041369800047
# Neural Net           0.40275990193825173
# AdaBoost             0.39660805944993494
# Naive Bayes          0.44830307209070713
# QDA                  0.4180897111775071