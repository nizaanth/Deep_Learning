import os
from os.path import join
from glob import glob

from natsort import natsorted

import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd

def read_mlrsnet(path_data, SEED):
    """Function to read image paths and labels from the MLRSNet dataset.

    Args:
        path_data (string): Path to the MLRSNet dataset.
        SEED (int): Seed to shuffle image paths.

    Returns:
        df_labels: DataFrame containing image paths and their corresponding labels.
        n_labels: Number of unique labels in the dataset.
    """
    
    # Define the directory containing the label CSV files
    data_dir = join(path_data, "labels")
    
    # Use glob to find all CSV files in the specified directory
    csv_files = glob(f"{data_dir}/*.csv")  

    # Initialize an empty list to store DataFrames from each CSV file
    all_dfs = []

    # Loop through each CSV file, read it into a DataFrame, and append to the list
    for filename in csv_files:
        df = pd.read_csv(filename)
        all_dfs.append(df)

    # Concatenate all DataFrames from the list into a single DataFrame
    df_labels = pd.concat(all_dfs, ignore_index=True) 

    # Define a function to extract column names with a value of 1 for each row,
    # except for the 'image' column. These columns represent the labels for the image.
    def get_ones(row):
        ones = [col for col in row.index if row[col] == 1 and col != 'image']
        return ones

    # Apply the function to each row in the DataFrame to create a 'labels' column
    df_labels['labels'] = df_labels.apply(get_ones, axis=1)

    # Create a DataFrame that counts the number of occurrences of each label
    class_count = pd.DataFrame(df_labels.sum(axis=0)).reset_index()
    class_count.columns = ["class", "count"]

    # Drop the first row (which corresponds to the 'image' column)
    class_count.drop(class_count.index[0], inplace=True)

    # Drop the last row, which contains the sum of the 'labels' column
    class_count.drop(class_count.index[-1], inplace=True)

    # Assuming df_labels contains an 'image' column with filenames like 'mobile_home_park_00003.jpg'
    # and path_data is the base directory path

    # Define a function to generate the full path
    def generate_path(row):
        # Extract the image name from the 'image' column
        image_name = row['image']
        
        # Split the image name on underscores and remove the last element (the number and extension)
        base_name = '_'.join(image_name.split('_')[:-1])
        
        # Construct the full path by concatenating path_data, base_name, and the original image name
        full_path = os.path.join(path_data, "Images", base_name, image_name)
        
        return full_path

    # Apply the function to each row in the DataFrame to create a new 'path_image' column
    df_labels['path_image'] = df_labels.apply(generate_path, axis=1)

    # Return the DataFrame with image paths and labels
    return df_labels, df_labels.shape[1] - 3

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
    y = df.drop(['image', 'labels', 'path_image'], axis=1).values

    # Train and test split. Use test_size here.
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=SEED,
                                                        )

    val_size_relative = val_size / (1 - test_size)
    # Train and validation split. Use val_size_relative here.
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=val_size_relative,
                                                      random_state=SEED,
                                                      )

    splits["x_train"] = x_train
    splits["y_train"] = y_train
    splits["x_val"] = x_val
    splits["y_val"] = y_val
    splits["x_test"] = x_test
    splits["y_test"] = y_test

    return splits
