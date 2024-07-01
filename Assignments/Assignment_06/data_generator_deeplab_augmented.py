from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
import random

from albumentations import (Compose, HorizontalFlip,
                            RandomRotate90, VerticalFlip,
                            ShiftScaleRotate)

def augmentation():
    return Compose([HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    RandomRotate90(p=0.5),
                    ShiftScaleRotate(shift_limit=0.01,
                                     scale_limit=0,
                                     rotate_limit=5, p=0.5),
                   ], p = 1)


class DataGenerator(keras.utils.Sequence):
  def __init__(self,
               batch_size,
               patch_size,
               step_size,
               list_top,
               list_gts,
               n_classes,
               colormap_gt=None,
               shuffle=True,
               augmentation=None):
    """Constructor

    Args:
        batch_size (int): number of patches per batch
        patch_size (int): size of the patch to be extracted
        step_size (int): stride for patch extraction
        list_top (list): list with paths of the true orthophotos
        list_gts (list): list with paths of the labels
        n_classes (int): number of classes
        colormap_gt (dict, optional): dictionary with the available
          colors and classes in the labels. Defaults to None.
        shuffle (bool, optional): If True, all patches are shuffled
          after each epoch. Defaults to True.
    """
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.step_size = step_size
    self.list_top = list_top
    self.list_gts = list_gts
    self.n_classes = n_classes
    self.colormap_gt = colormap_gt
    self.shuffle= shuffle
    self.path_coords = self.__get_path_coords()
    self.n_patchs = len(self.path_coords)
    self.augmentation = augmentation  # Flag to control augmentation

    # Define augmentation pipeline using albumentations if augmentation is True
    self.augmentor = augmentation() if augmentation else None

  def __get_path_coords(self):
    """Get a list with information about the patches to be extracted.
      The image path, label path, and X and Y coordinates are storaged.

    Returns:
        list: list with image path, label path, and X and Y
          coordinates of all patches to be extracted.
    """
    path_coords = []
    for path_image, path_label in zip(self.list_top,self.list_gts):
      width, height = Image.open(path_label).size
      for y in range(0, height, self.step_size):
        for x in range(0, width, self.step_size):
          if (x + self.patch_size) > width:
            x = width - self.patch_size
          if (y + self.patch_size) > height:
            y = height - self.patch_size

          path_coords.append((path_image, path_label, (x,y)))
    return path_coords

  def on_epoch_end(self):
    """Executes at the end of an epoch.
    """
    if self.shuffle:
      random.shuffle(self.path_coords)

  def __len__(self):
    """Computes the number of batches.

    Returns:
        int: number of batches
    """
    return np.ceil(self.n_patchs/self.batch_size).astype("int")

  def __get_patch_image(self, 
                        path_image, 
                        x, 
                        y):
    """Function to read an image and preprocess it.

    Args:
        path_image (string): image path
        x (int): coordinate X of the patch to be extracted
        y (int): coordinate Y of the patch to be extracted

    Returns:
        np.array: image array with shape [batch_size,width,height,channels]
    """
    # Reading the RGB image
    data_image = Image.open(path_image)
    data_image = data_image.crop((x, y,
                                  x + self.patch_size, y + self.patch_size))
    data_image = np.asarray(data_image).astype("float32")

    return data_image/255.0

  def __get_patch_mask(self, 
                       path, 
                       x, 
                       y):
    """Function to read a label and preprocess it.

    Args:
        path (string): label path
        x (int): coordinate X of the patch to be extracted
        y (int): coordinate Y of the patch to be extracted

    Returns:
        np.array: image array with shape [batch_size,width,height,channels]
    """
    # Reading the mask
    data_mask = Image.open(path)
    data_mask = data_mask.crop((x, y,
                                x + self.patch_size, y + self.patch_size))
    data_mask = np.asarray(data_mask)
    # Convert RGB image [H x W x 3] to label image [H x W x 1]
    data_mask_idx = np.zeros((data_mask.shape[0],
                              data_mask.shape[1]), dtype=np.uint8)
    for key in self.colormap_gt:
      class_id = self.colormap_gt[key]
      key = np.array(key)
      # Assign an index to each color based on the colormap
      # (255,255,255) : 0
      # (0  ,0  ,255) : 1,...
      data_mask_idx[(data_mask[:,:,0] == key[0]) &
                    (data_mask[:,:,1] == key[1]) &
                    (data_mask[:,:,2] == key[2])] = class_id
    return data_mask_idx

  def __getitem__(self, 
                  idx):
    """Function that provides a batch of data.

    Args:
        idx (int): batch index

    Returns:
        tuple: tuple of np.arrays with the image and its label.
    """
    i = idx * self.batch_size

    current_batch_size = self.batch_size
    if (idx+1) == self.__len__():
      current_batch_size = len(self.path_coords[i:])

    # Batch of coordinates
    batch_path_coords = self.path_coords[i : i + current_batch_size]

    x = np.zeros((current_batch_size,
                  self.patch_size,
                  self.patch_size,
                  3),
                  dtype=np.float32)

    y = np.zeros((current_batch_size,
                  self.patch_size,
                  self.patch_size,
                  self.n_classes),
                  dtype=np.float32)

    for j, (path_image, path_label, (x_pos, y_pos)) in enumerate(batch_path_coords):
      # Get an individual image and its corresponding label
      x_sample = self.__get_patch_image(path_image, x_pos, y_pos)
      y_sample = self.__get_patch_mask(path_label, x_pos, y_pos)

      # If there are augmentation transformations, apply them
      if self.augmentor: 
         augmented = self.augmentor(image=x_sample, mask=y_sample)
         x_sample = augmented["image"]
         y_sample = augmented["mask"]

      # Convert labels to one hot encoding
      y_sample = to_categorical(y_sample, self.n_classes)

      x[j,...] = x_sample
      y[j,...] = y_sample
    return x, y