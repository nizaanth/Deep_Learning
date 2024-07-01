from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import cv2
import numpy as np


from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import cv2
import numpy as np


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 path_images,
                 labels,
                 batch_size,
                 n_classes,
                 target_size=256,
                 shuffle=True):
        """Constructor.

        Args:
            path_images (np.array): array with images path
            labels (np.array): array with the corresponding labels
            batch_size (int): number of samples per batch
            n_classes (int): number of classes
            target_size (int): size of each image in a batch.
              Defaults to 256.
            shuffle (bool, optional): If True, all samples are shuffled
              after each epoch. Defaults to True.
        """
        self.path_images = path_images
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.target_size = target_size
        self.shuffle = shuffle

    def on_epoch_end(self):
        """Executes at the end of an epoch.
        """
        if self.shuffle:
            self.path_images, self.labels = shuffle(self.path_images,
                                                    self.labels)

    def __len__(self):
        """Computes the number of batches.

        Returns:
            int: number of batches
        """
        return np.ceil(self.path_images.shape[0] / self.batch_size).astype("int")

    def __get_image(self, path_image):
        """Function to read an image and preprocess it.

        Args:
            path_image (string): image path

        Returns:
            np.array: image array with shape [batch_size,width,height,channels]
        """
        # Read the image using OpenCV as the image format is JPG.
        # Remember to include a dimension for the batches.
        # read image using cv2
        x_sample = cv2.imread(path_image)
        # resize image using cv2
        x_sample = cv2.resize(x_sample, (self.target_size, self.target_size))
        # convert the image from BGR to RGB
        x_sample = cv2.cvtColor(x_sample, cv2.COLOR_BGR2RGB)
        # increase one dimension for the batch_size
        x_sample = np.expand_dims(x_sample, axis=0)
        # convert to float and normalize
        x_sample = x_sample.astype("float") / 255.0

        return x_sample

    def __get_label(self, label):
        """Function to read a label and preprocess it.

        Args:
            label (int): image label for classification

        Returns:
            np.array: array with shape [batch_size,n_classes]
        """
        # convert label to one-hot encoding
        y_sample = to_categorical(label, num_classes=self.n_classes)
        return y_sample

    def __getitem__(self, idx):
        """Function that provides a batch of data.

        Args:
            idx (int): batch index

        Returns:
            tuple: tuple of np.arrays with the image and its label.
        """
        i = idx * self.batch_size

        current_batch_size = self.batch_size
        if (idx + 1) == self.__len__():
            current_batch_size = len(self.path_images[i:])

        batch_images = self.path_images[i: i + current_batch_size]
        batch_labels = self.labels[i: i + current_batch_size]

        # [batch_size, width, height, 3]
        x = np.zeros((current_batch_size,
                      self.target_size,
                      self.target_size,
                      3),
                     dtype=np.float32)
        # [batch_size, width, height, n_classes]
        y = np.zeros((current_batch_size,
                      self.n_classes),
                     dtype=np.float32)

        # read data
        for j, (path_image, label) in enumerate(zip(batch_images, batch_labels)):
            # Reading each image
            x_sample = self.__get_image(path_image)
            # Get the label
            y_sample = self.__get_label(label)

            x[j, ...] = x_sample
            y[j, ...] = y_sample

        return x, y
