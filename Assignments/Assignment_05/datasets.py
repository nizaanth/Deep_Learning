import os
from os.path import join
from glob import glob

from natsort import natsorted

import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd


def read_eurosat(path_data, SEED):
    """Function to read images paths of the EuroSAT dataset

    Args:
      path_data (string): path to the EuroSAT dataset.
      SEED (int): seed to shuffle image paths.

    Returns:
      df: dataframe with information about the image paths and their
        corresponding classes as string and int.
      n_classes: number of classes available in the dataset.
    """
    # List with all images in the folder
    list_img = glob(join(path_data, "**", "*.jpg"), recursive=True)
    list_img = natsorted(list_img, key=lambda y: y.lower())

    # Dataframe for better management of image paths
    df = pd.DataFrame(list_img, columns=["path_image"])

    # Getting class name from filename path
    # Example: .\EuroSAT_RGB\AnnualCrop\AnnualCrop_1.jpg	
    #          class = AnnualCrop
    df["class_str"] = df["path_image"].apply(lambda x: x.split(os.sep)[-2])

    classes = np.unique(df["class_str"].values)
    n_classes = len(classes)

    classes_int = np.arange(len(classes))
    classes_dict = dict(zip(classes, classes_int))

    # Applying the dictionary to the column "class"
    df["class_int"] = df["class_str"].apply(lambda x: classes_dict[x])

    # Shuffle the dataframe rows without keeping the old index
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df, n_classes



def train_val_test_split(df, val_size, test_size, SEED):
    """Function to create three disjoint sets for train, validation
    and test.

    Args:
      df (dataframe): pandas dataframe with information about the
        images paths and their corresponding class.
      val_size (float): percentage of the dataset used for validation.
      test_size (float): percentage of the dataset used for test.
      SEED (int): seed to split the dataset.

    Returns:
      splits: dictionary with the images and classes for train,
        validation and test.
    """
    splits = dict()

    # Extracting features and labels
    x = df["path_image"].values
    y = df["class_int"].values

    # TODO: Train and test split. Use test_size here.
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=SEED,
                                                        stratify=y)

    val_size_relative = val_size / (1 - test_size)
    # TODO: Train and validation split. Use val_size_relative here.
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=val_size_relative,
                                                      random_state=SEED,
                                                      stratify=y_train)

    splits["x_train"] = x_train
    splits["y_train"] = y_train
    splits["x_val"] = x_val
    splits["y_val"] = y_val
    splits["x_test"] = x_test
    splits["y_test"] = y_test

    return splits
