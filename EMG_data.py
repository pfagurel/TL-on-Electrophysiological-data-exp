import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


##functions for extracting sEMG features
def rms(data):  ##root mean square
    return np.sqrt(np.mean(data ** 2, axis=0))


def SSI(data):  ##Simple Square Integral
    return np.sum(data ** 2, axis=0)


def abs_diffs_signal(data):  ##absolute differential signal
    return np.sum(np.abs(np.diff(data, axis=0)), axis=0)


##function for returning an estimator class name
def print_estimator_name(estimator):
    return estimator.__class__.__name__


def get_dataset():
    Input_path = 'EMG-data.csv'

    dataset = pd.read_csv(Input_path)

    ##drop gesture 0 because it offers no information due to its unmarked nature
    index_numbers_1 = dataset[dataset["class"] == 0].index
    dataset.drop(index_numbers_1, inplace=True)
    ##drop gesture 7 because it offers no information due to it being performed
    ##by just two out of 36 patients
    index_numbers_2 = dataset[dataset["class"] == 7].index
    dataset.drop(index_numbers_2, inplace=True)

    return dataset


def get_data(_dataset, subject):
    dataset_subject = _dataset[_dataset["label"] == subject]
    dataset = _dataset[_dataset["label"] != subject]
    temp_dataset_subject_class1_time = dataset_subject["time"]

    dataset_subject_part1 = None
    dataset_subject_part2 = None
    i = 0
    prec_time_i = 0
    for time_i in temp_dataset_subject_class1_time:
        if time_i < prec_time_i:
            dataset_subject_part1 = dataset_subject.iloc[:i, :]
            dataset_subject_part2 = dataset_subject.iloc[i:, :]
            break
        prec_time_i = time_i
        i += 1

    dataset = dataset.drop(columns=["time"])
    dataset_subject_part1 = dataset_subject_part1.drop(columns=["time"])
    dataset_subject_part2 = dataset_subject_part2.drop(columns=["time"])

    dataset = dataset.groupby(['label', 'class'])
    dataset_subject_part1 = dataset_subject_part1.groupby(['label', 'class'])
    dataset_subject_part2 = dataset_subject_part2.groupby(['label', 'class'])
    ##tabulating the aggregated sEMG features
    dataset_subject_part1 = dataset_subject_part1.agg(['min', 'max', rms, SSI, abs_diffs_signal, np.ptp])
    dataset_subject_part2 = dataset_subject_part2.agg(['min', 'max', rms, SSI, abs_diffs_signal, np.ptp])
    dataset = dataset.agg(['min', 'max', rms, SSI, abs_diffs_signal, np.ptp])

    dataset = dataset.reset_index()
    dataset_subject_part1 = dataset_subject_part1.reset_index()
    dataset_subject_part2 = dataset_subject_part2.reset_index()

    labels_subject_part1 = dataset_subject_part1["class"]
    labels_subject_part2 = dataset_subject_part2["class"]
    labels_data = dataset["class"]

    features_subject_part1 = dataset_subject_part1.drop(columns=["label", "class"], level=0)
    features_subject_part2 = dataset_subject_part2.drop(columns=["label", "class"], level=0)
    features_data = dataset.drop(columns=["label", "class"], level=0)

    return (features_data, labels_data, features_subject_part1, labels_subject_part1,
            features_subject_part2, labels_subject_part2)
