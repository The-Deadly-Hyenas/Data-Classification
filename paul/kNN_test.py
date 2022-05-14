import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN
from joblib import load
import pathlib

# load the data and reset index of dataframe
df: pd.DataFrame = pd.read_pickle(
    "../training_dataset_task3/task_3_training_e8da4715deef7d56_f8b7378_pandas.pkl").reset_index()

# get only the low and mid level features + segment_id
X = df.loc[:, "essentia_dissonance_mean":"mirtoolbox_roughness_pct_90"]
# target value
y = df["quadrant"]

# preprocess dataset
X_std = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_std, columns=X.columns)

# add the highlevel features generated from SVMs if generated is True
generate_data = True
if generate_data:

    trainset_path = list(pathlib.Path(".").glob("*high*"))
    if False and not trainset_path:

        training_data = X.copy()
        model_paths = [path for path in pathlib.Path("../patrick/models").iterdir() if path.is_file()]

        for model_path in model_paths:
            model = load(model_path)
            feature = model_path.name
            pred = model.predict(X.values)
            training_data.insert(0, feature, pred)
            print(feature + " generated")

        training_data.to_pickle("training_data_with_highlevel.pkl")

    training_data = pd.read_pickle(trainset_path[0])  # reloading data to make sure it was saved
    X = training_data

# add segment_id to training data for doing the cross validation splits
X["segment_id"] = df["segment_id"]

# remove segment_id 26
seg_26_indices = (X["segment_id"] == 26)
X_test = X[seg_26_indices].drop(["segment_id"], axis=1)
y_test = y[seg_26_indices]

X_train = X.drop(X[seg_26_indices].index, axis=0).reset_index(drop=True)
y_train = y.drop(X[seg_26_indices].index, axis=0)

# https://imbalanced-learn.org/stable/combine.html
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# split the data according to segment_id
# store the splits as tuple (train indices, test_indices)
# 2 segments for test, the rest for training (not including segment 26)
cv = []

for i in range(24):
    train_indices = X_resampled[~X_resampled["segment_id"].isin([i, i + 1])].index.to_list()
    test_indices = X_resampled[X_resampled["segment_id"].isin([i, i + 1])].index.to_list()
    cv.append((train_indices, test_indices))

# remove the segment_id as we don't want it in the training data
X_resampled = X_resampled.drop(["segment_id"], axis=1)

# do a grid search for best params
params = {"n_neighbors": [81],
          "weights": ["distance"],
          "algorithm": ["auto"]}

gs_cv = GridSearchCV(KNeighborsClassifier(), params, cv=cv, n_jobs=-1)
gs_cv.fit(X_resampled, y_resampled)

best_params = gs_cv.best_params_

print("Best params:")
print(best_params)
print("Best score:")
print(gs_cv.best_score_)

# get score on hold out test set
X_test = X[X["segment_id"] == 26].drop(["segment_id"], axis=1)
y_test = y[X["segment_id"] == 26]

test_score = gs_cv.score(X_test, y_test)

print("Test score:")
print(test_score)
