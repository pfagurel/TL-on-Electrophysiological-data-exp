import numpy as np
import pandas

import EMG_data as EMG_data
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from adapt.parameter_based import TransferForestClassifier
import matplotlib.pyplot as plt
import time


def compute_accuracy(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees):
    source_model = RandomForestClassifier(n_estimators=nb_trees)

    start = time.time()
    source_model.fit(Xs, ys)
    end = time.time()
    #print(end - start, "SOURCE seconds")

    model = TransferForestClassifier(source_model)

    start = time.time()
    model.fit(X_transfer.to_numpy(), y_transfer.to_numpy()-1)
    end = time.time()
    #print(end - start, "TransferForestClassifier seconds")


    #print("fit score:", model.score(Xt, yt))
    return (accuracy_score(yt, source_model.predict(Xt)), accuracy_score(yt, model.predict(Xt)))


def compute_mean_accuracy(dataset, nb_subjects, nb_trees):

    accuracy_subject_source_vec = np.array([None]*nb_subjects)
    accuracy_subject_tl_vec = np.array([None]*nb_subjects)

    for subject in range(1, nb_subjects+1):
        Xs, ys, X_transfer, y_transfer, Xt, yt = EMG_data.get_data(dataset, subject)

        accuracy_subject_source_vec[subject-1], accuracy_subject_tl_vec[subject - 1]\
            = compute_accuracy(Xs, ys, X_transfer, y_transfer, Xt, yt, nb_trees)

    return (accuracy_subject_source_vec.mean(), accuracy_subject_tl_vec.mean())


dataset = EMG_data.get_dataset()
nb_subjects = 36
nb_trees_vec = (10, 50, 100, 300, 500, 1000)

accuracy_per_trees_source = pandas.array([None] * len(nb_trees_vec))
accuracy_per_trees_tl = pandas.array([None] * len(nb_trees_vec))

for i in range(len(nb_trees_vec)):
    accuracy_per_trees_source[i], accuracy_per_trees_tl[i] =\
    compute_mean_accuracy(dataset, nb_subjects, nb_trees_vec[i])


pandas.DataFrame({"nb_trees": nb_trees_vec, "Source only accuracy": accuracy_per_trees_source,
                  "TransferForestClassifier accuracy": accuracy_per_trees_tl}).plot(x="nb_trees", marker='o')
plt.show()
