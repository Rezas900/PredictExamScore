import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def ShuffleSpliter(dataset, bins, labels, targetColumn,
                   n_split=1, test_size=0.2,
                   random_state=42):
    dataset["Temporary"] = pd.cut(dataset[targetColumn]
                                  , bins=bins
                                  , labels=labels)
    split = StratifiedShuffleSplit(n_splits=n_split,
                                   test_size=test_size,
                                   random_state=random_state)
    for train_index, test_index in split.split(dataset, dataset["Temporary"]):
        train_ = dataset.loc[train_index]
        test_ = dataset.loc[test_index]
    for set in [train_, test_]:
        set.drop(["Temporary"], axis=1, inplace=True)
    return train_, test_
