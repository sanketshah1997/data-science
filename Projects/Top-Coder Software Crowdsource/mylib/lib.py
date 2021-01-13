import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
def split_train_test(data, ratio=0.2):
    """
    Funtion for splitting the dataset into train and test.
    Args:
        data (pandas dataFrame, required): [data that will be converted into train and test]
        ratio (float, optional): [test ratio]. Defaults to 0.2.
    """
    np.random.seed(39)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_set_size]

def strata_train_test(data, ratio=0.2, strata_col = 'strata_col'):
    """
    Funtion for splitting the dataset into train and test using stratified split.
    Args:
        data (pandas dataFrame, required): [data that will be converted into train and test]
        ratio (float, optional): [test ratio]. Defaults to 0.2.
        strata_col(str, default = 'strata_col'). column name to be used for stratified sampling.
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(data[strata_col],strata_col):
        strata_train = data.loc[train_idx]
        strata_test = data.loc[test_idx]
    return strata_train, strata_test