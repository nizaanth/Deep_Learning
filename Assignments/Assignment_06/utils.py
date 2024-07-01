from glob import glob
from os.path import join,splitext,basename
from natsort import natsorted
import random
import tifffile
import numpy as np
import matplotlib.pyplot as plt

def read_vaihingen(path_dataset):
  """Read images and labels paths of the 
      Vaihingen dataset

  Args:
      path_dataset (string): Path to the folder 
      with the Vaihingen dataset

  Returns:
      tuple: two list with the path of the 
      images and labels
  """
  path_top = join(path_dataset,"top")
  path_gts = join(path_dataset,"gts")

  # List with all TIF images in each folder
  list_top = glob(join(path_top,"*.tif"))
  list_top = natsorted(list_top, key=lambda y: y.lower())

  list_gts = glob(join(path_gts,"*.tif"))
  list_gts = natsorted(list_gts, key=lambda y: y.lower())

  return list_top, list_gts
  
def train_val_test_vaihingen(list_top, 
                             list_gts,
                             val_size=0.25,
                             seed=42):
  """Function to generate the train, validation and test 
  sets of the Vaihingen dataset

  Args:
      list_top (list): list with the paths of the images
      list_gts (list): list with the paths of the labels
      val_size (float, optional): Proportion to be used
        for validation. Defaults to 0.25.
      seed (int, optional): Seed to shuffle train samples
        before split. Defaults to 42.

  Returns:
      splits : dictionary with the paths of the images and
        labels for train, validation, and test
  """
  # Based on the benckmark rules, 
  # these are the areas for train and test
  train_ids = [1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37]
  test_ids = [2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38]

  # Initialization of train, val and test sets
  list_top_train, list_top_val, list_top_test = ([] for i in range(3))
  list_gts_train, list_gts_val, list_gts_test = ([] for i in range(3))

  for top,gts in zip(list_top,list_gts):
    # Get the area ID from the filename
    # Example: ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area1.tif
    area_id = int(splitext(basename(top))[0].split("area")[-1])
    # Train set
    if area_id in train_ids:
      list_top_train.append(top)
      list_gts_train.append(gts)
    # Test set
    elif area_id in test_ids:
      list_top_test.append(top)
      list_gts_test.append(gts)
  
  # shuffle train lists
  temp = list(zip(list_top_train,list_gts_train))
  random.seed(seed)
  random.shuffle(temp)
  list_top_train,list_gts_train = zip(*temp)

  # take val_size as validation set
  n_val = int(val_size*len(list_top_train))

  list_top_val = list_top_train[:n_val]
  list_top_train = list_top_train[n_val:]

  list_gts_val = list_gts_train[:n_val]
  list_gts_train = list_gts_train[n_val:]

  splits = dict()
  splits["top_train"] = list_top_train
  splits["gts_train"] = list_gts_train
  splits["top_val"] = list_top_val
  splits["gts_val"] = list_gts_val
  splits["top_test"] = list_top_test
  splits["gts_test"] = list_gts_test

  return splits

def print_info(list_top, list_gts):
  """Function to print information of the Vaihingen dataset
   about number of classes

  Args:
      list_top (list): list with the paths of the true orthophotos
      list_gts (list): list with the paths of the laels
  """
  for top,gts in zip(list_top,list_gts):
    gt_temp = tifffile.imread(gts)
    n_classes = len(np.unique(gt_temp.reshape(-1,3), axis=0))
    print("TOP:{} , GT:{} , N:{}".format(basename(top),
                                          basename(gts),
                                          n_classes))

def show_batch(img, label, colormap):
  """Function to show the content in a batch

  Args:
      img (np.array): array with image data from
        a batch
      label (np.array): array with label data from
        a batch
      colormap (dict): dictionary with the colormap
        and labels for visualization
  """
  for x_i,y_i in zip(img,label):
    # For visualization: [0,1] => [0,255]
    x_i = x_i*255
    # Apply argmax to convert from one hot to integers
    y_i = np.argmax(y_i, axis=-1)
    # Convert each class to a color given a colormap
    y_i_rgb = np.zeros((y_i.shape[0],
                        y_i.shape[1],
                        3),
                       dtype=np.uint8)
    for key,value in colormap.items():
      y_i_rgb[y_i==value] = key
    # Visualization of RGB image and its ground truth or reference
    plt.figure(figsize=(5,5))
    plt.subplot(1,2,1)
    plt.imshow(x_i.astype("uint8"))
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(y_i_rgb, cmap="jet")
    plt.title("Ground Truth")
    plt.axis("off")
    plt.show()
