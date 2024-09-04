# %%
# Management of files
import os
from os.path import join,basename
from glob import glob
from natsort import natsorted
import pandas as pd
import numpy as np
from tqdm import tqdm



import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split

# Monitor training
import wandb
from wandb.integration.keras import WandbMetricsLogger

# External files with functions to load the dataset,
# create a CNN model, and a data generator.
from importlib import reload
import datasets_mlrsnet
import data_generator
import new_metrics

# Useful to reload modified external files without need
# of restarting the kernel. Just run again this cell.
reload(datasets_mlrsnet)
reload(data_generator)
reload(new_metrics)

from datasets_mlrsnet import *
from data_generator import *
from new_metrics import *

# %%
SEED = 42
BATCH_SIZE = 32
TARGET_SIZE = 256
WANDB_KEY = '587e039f45054814160e93a4c0fbbe9a6349b518'

# %%

#wandb.login(key=WANDB_KEY)

# %%
PROJECT_DIR = "." # os.getcwd()

# %%
"""
### **Reading the dataset**
"""

# %%
"""
The function read_mlrsnet is implemented in the datasets.py file. The output of this function are a dataframe with information about the image paths and their corresponding classes, and the number of classes.
"""

# %%
path_data = join(PROJECT_DIR, "MLRSNet-master")

df, n_labels = read_mlrsnet(path_data=path_data, SEED=SEED)

# Shuffle the DataFrame
shuffled_df = df.sample(frac=1, random_state=1).reset_index(drop=True)

# Cut it down to just 10,000 entries
subset_df = shuffled_df.iloc[:10000]


# %%
"""
### **Train, Validation and Test sets**
"""

# %%
"""
Create **three disjoint** sets: `train`, `validation` and `test`.

Use the following proportions:
- `train`: 20%
- `validation`: 10%
- `test`: 70%

Remember to use **stratified sampling** and the given `SEED` for the splits.
"""

# %%
splits = train_val_test_split(subset_df,
                              val_size=0.1,
                              test_size=0.7,
                              SEED=SEED)

x_train = splits["x_train"]
y_train = splits["y_train"]
x_val = splits["x_val"]
y_val = splits["y_val"]
x_test = splits["x_test"]
y_test = splits["y_test"]

# %%
"""
#### **Class distribution**

For **sanity check**, verify the **class distribution** of each set: `train`, `validation` and `test`.
"""

# %%
x_train

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen_train = DataGenerator(path_images=x_train,
                               labels=y_train,
                               batch_size=BATCH_SIZE,
                               n_classes=n_labels,
                               target_size=TARGET_SIZE,
                               shuffle=True)

data_gen_val = DataGenerator(path_images=x_val,
                             labels=y_val,
                             batch_size=BATCH_SIZE,
                             n_classes=n_labels,
                             target_size=TARGET_SIZE,
                             shuffle=False)

# For sanity check, let's see the generator's output
for i, (x, y) in enumerate(data_gen_train):
    print(i, x.shape, y.shape)

# %%
"""
## **Custom loss function**
"""
def custom_loss(t, y_pred):
    """
    Custom loss function for multi-class classification with binary cross-entropy.

    Args:
        t: Tensor of true labels (shape: [batch_size, num_classes])
        y_pred: Tensor of predicted probabilities (shape: [batch_size, num_classes])
    
    Returns:
        loss: Scalar tensor representing the computed loss
    """
 # Get the dimensions of the input tensor
    m = tf.cast(tf.shape(t)[0], tf.float32)  # number of samples as float
    q = tf.cast(tf.shape(t)[1], tf.float32)  # number of classes as float

    # Clip predictions to avoid log(0)
    #y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    # Compute the loss
    loss_sum = t * tf.math.log(y_pred) + (1 - t) * tf.math.log(1 - y_pred)
    loss_sum = -tf.reduce_sum(loss_sum, axis=1)  # Sum over classes for each sample
    loss = tf.reduce_mean(loss_sum) / (m * q)  # Average loss over the batch and number of classes
    
    return loss


def freeze_up_to(model, freeze_layer_name):
  """Function to freeze some layers of the model

  Args:
      model (keras.Model): a keras.Model
      freeze_layer_name (str): layer name of "model". All layers up
        to this layer will be freezed.

  Returns:
      keras.Model: a keras.Model with some layers freezed.
  """
  # Getting layer number based on layer name
  for id_layer, layer in enumerate(model.layers):
    if layer.name == freeze_layer_name:
      layer_number = id_layer
      break

  # Froze layers
  for layer in model.layers[:layer_number]:
    layer.trainable = False
  return model

# %%
# VGG16 base model initialization without the top layers
vgg16_base = VGG16(include_top=False,
                   weights='imagenet',  # Use None for random initialization
                   input_shape=(TARGET_SIZE, TARGET_SIZE, 3))

#vgg16_base = freeze_up_to(vgg16_base, "block3_conv3")
vgg16_base.trainable = True
# Print the summary of the base model
vgg16_base.summary()


# Define the input layer
input = Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))

x = preprocess_input(input)


# Pass input through VGG16 base
x = vgg16_base(x)

# Flatten the output of VGG16 base
x = Flatten()(x)


# Add fully connected layers
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
output = Dense(n_labels, activation="sigmoid")(x)

# Define the model
model = Model(inputs=input, outputs=output)

# Print the summary of the complete model
model.summary()
# %%
#Mean Average Precision
#mAP = tfr.keras.metrics.MeanAveragePrecisionMetric()

# %%
"""
### **Model callbacks**
"""

# %%
"""
Define the callbacks for training:
"""

# %%
# Callbacks
cb_autosave = ModelCheckpoint("ResNet152_model_01.h5",
                              mode="min",
                              save_best_only=True,
                              monitor= "val_loss",  # Monitor based on validation binary accuracy
                              verbose=1)

cb_early_stop = EarlyStopping(patience=20,
                              verbose=1,
                              mode="min",
                              restore_best_weights=True,
                              monitor= "val_loss")  # Monitor based on validation binary accuracy



callbacks = [cb_autosave, cb_early_stop]

# %%

# Compile the model with Hamming score as a metric
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss= custom_loss,                                                                                                                                                                                                                                                                                   
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5),hamming_score_metric, sampled_f1_score_metric, precision_metric,
                       recall_metric])

# Train the model
history = model.fit(data_gen_train, epochs=10, validation_data=data_gen_val, callbacks=callbacks)

# Save the trained model as a .h5 file
model.save('vgg16_trained_model_07.h5')


# %%


#model = tf.keras.models.load_model('vgg16_trained_model.h5')

# %%
data_gen_test = DataGenerator(path_images=x_test,
                             labels=y_test,
                             batch_size=BATCH_SIZE,
                             n_classes=n_labels,
                             target_size=TARGET_SIZE,
                             shuffle=False)


# %%

print("Train:")
scores_train = model.evaluate(data_gen_train)
print("Validation:")
scores_val = model.evaluate(data_gen_val)
print("Test:")
scores_test = model.evaluate(data_gen_test)


# Visualisng the predictions on the test dataset

# In[ ]:


# Define your class labels based on the filenames
class_labels = [
    'airplane', 'airport', 'bareland', 'baseball_diamond', 'basketball_court', 'beach',
    'bridge', 'chaparral', 'cloud', 'commercial_area', 'dense_residential_area', 'desert',
    'eroded_farmland', 'farmland', 'forest', 'freeway', 'golf_course', 'ground_track_field',
    'harbor&port', 'industrial_area', 'intersection', 'island', 'lake', 'meadow',
    'mobile_home_park', 'mountain', 'overpass', 'park', 'parking_lot', 'parkway',
    'railway', 'railway_station', 'river', 'roundabout', 'shipping_yard', 'snowberg',
    'sparse_residential_area', 'stadium', 'storage_tank', 'swimming_pool', 'tennis_court',
    'terrace', 'transmission_tower', 'vegetable_greenhouse', 'wetland', 'wind_turbine'
]


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Function to visualize the results
def visualize_predictions(model, data_gen_test, class_labels, num_images=3):
    """
    Visualizes a number of images along with their true and predicted labels.

    Parameters:
    - model: Trained TensorFlow model
    - data_gen_test: Data generator for the test set
    - class_labels: List of class labels corresponding to the dataset
    - num_images: Number of images to visualize
    """
    # Get a batch of images and their corresponding labels
    images, true_labels = next(iter(data_gen_test))

    # Make predictions on the images
    predictions = model.predict(images)

    # Convert binary predictions to class labels
    pred_labels = (predictions >= 0.5).astype(int)

    # Set the number of rows equal to the number of images
    rows = num_images  
    cols = 1  # One column since we want one image per row

    # Visualize the images, true labels, and predicted labels
    plt.figure(figsize=(10, 5 * rows))  # Adjust the figure size according to the number of images
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)  # Create a grid of subplots
        plt.imshow(images[i].astype("uint8"))
        plt.axis('off')

        # Safe handling of true and predicted labels, checking for NaN
        true_label_indices = np.where(~np.isnan(true_labels[i]) & (true_labels[i] == 1))[0]
        pred_label_indices = np.where(~np.isnan(pred_labels[i]) & (pred_labels[i] == 1))[0]

        # Get valid true labels safely
        valid_true_labels = [class_labels[idx] for idx in true_label_indices if 0 <= idx < len(class_labels)]
        true_label_text = ', '.join(valid_true_labels) if valid_true_labels else 'No true labels'

        # Get valid predicted labels safely
        valid_pred_labels = [class_labels[idx] for idx in pred_label_indices if 0 <= idx < len(class_labels)]
        pred_label_text = ', '.join(valid_pred_labels) if valid_pred_labels else 'No predicted labels'

        # Adjust title size and spacing
        plt.title(f'True: {true_label_text}\nPred: {pred_label_text}', fontsize=8, pad=10)  # Decreased font size

    plt.tight_layout(pad=3.0)  # Adjust spacing between plots

    plt.savefig('output_plot_3.png')
    plt.show()

# Call the function to visualize predictions
visualize_predictions(model, data_gen_test, class_labels, num_images=5)


# Generate predictions
y_pred_probs = model.predict(data_gen_test)

# For binary predictions at a threshold of 0.5
y_pred = (y_pred_probs >= 0.5).astype(int)  # Convert probabilities to binary predictions
y_true = data_gen_test.labels  # Assuming data_gen_test.labels gives the true labels

# Calculate F1 score (samples average)
f1 = f1_score(y_true, y_pred, average='samples', zero_division=0) 

print(f"F1 Score (samples): {round(f1*100,4)}")

# Calculate Mean Average Precision (mAP)
# If y_true is multi-label, average_precision_score can handle it directly
mAP = average_precision_score(y_true, y_pred_probs, average='samples')  # Use 'macro' for averaging

print(f"Mean Average Precision (mAP): {round(mAP*100, 4)}")