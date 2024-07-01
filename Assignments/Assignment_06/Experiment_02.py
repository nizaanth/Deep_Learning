# Management of files
import os
from os.path import exists, join

# Tensorflow and Keras
from tensorflow.keras.callbacks import ModelCheckpoint, \
                                       EarlyStopping
from tensorflow.keras.metrics import OneHotMeanIoU, OneHotIoU, MeanIoU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

# Monitor training
import wandb
from wandb.integration.keras import WandbMetricsLogger



# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

# Working with arrays
import numpy as np

# External files with functions to load the dataset,
# create a CNN model, and a data generator.
from importlib import reload
import utils
import models_deeplab
import models_unet
import data_generator
import losses
# Useful to reload modified external files without need
# of restarting the kernel. Just run again this cell.
reload(utils)
reload(models_deeplab)
reload(models_unet)
reload(data_generator)
reload(losses)

from utils import *
from models_deeplab import *
from models_unet import *
from data_generator import *
from losses import *


### VARIABLES ###

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

WANDB_KEY = "587e039f45054814160e93a4c0fbbe9a6349b518"
PROJECT_DIR = "." # os.getcwd()
SEED = 42
color2index = {(255,255,255) : 0,
               (0  ,0  ,255) : 1,
               (0  ,255,255) : 2,
               (0  ,255,0  ) : 3,
               (255,255,0  ) : 4,
               (255,0  ,0  ) : 5
               }
n_classes = len(color2index)

BATCH_SIZE = 8
PATCH_SIZE = 256
STEP_SIZE = 256
EPOCHS = 200

### READING THE DATASET ###

# Path to the dataset folder in Google Drive
DATA_PATH = join(PROJECT_DIR, "ISPRS_semantic_labeling_Vaihingen")

list_top, list_gts = read_vaihingen(DATA_PATH)


### TEST, TRAIN, VALIDATION SET ####

splits = train_val_test_vaihingen(list_top,
                                  list_gts,
                                  val_size=0.25,
                                  seed=SEED)
# Train
print("Train:")
print_info(splits["top_train"],splits["gts_train"])

# Validation
print("\nValidation:")
print_info(splits["top_val"],splits["gts_val"])

# Test
print("\nTest:")
print_info(splits["top_test"],splits["gts_test"])

### DATA GENERATOR ###
data_gen_train = DataGenerator(batch_size=BATCH_SIZE,
                               patch_size=PATCH_SIZE,
                               step_size=STEP_SIZE,
                               list_top=splits["top_train"],
                               list_gts=splits["gts_train"],
                               n_classes=n_classes,
                               colormap_gt=color2index
                               )

data_gen_val = DataGenerator(batch_size=BATCH_SIZE,
                             patch_size=PATCH_SIZE,
                             step_size=STEP_SIZE,
                             list_top=splits["top_val"],
                             list_gts=splits["gts_val"],
                             n_classes=n_classes,
                             colormap_gt=color2index
                             )

print("Number of patches for training: {}".format(len(data_gen_train)*BATCH_SIZE))

print("\nNumber of patches for validation: {}".format(len(data_gen_val)*BATCH_SIZE))


a, b = data_gen_train[0]
a.shape, b.shape
imgs, labels = data_gen_train[0]
show_batch(imgs, labels, color2index)



### MODEL - U-Net Compilation ###

unet = get_unet(img_size=PATCH_SIZE,
                n_classes=n_classes)


unet.compile(optimizer=Adam(),
             loss=categorical_focal_loss(gamma=2),
             metrics=["accuracy",
                      OneHotMeanIoU(num_classes=n_classes,name="miou")
            ])


### MODEL - U-Net Callbacks ###
# Callbacks

wandb.login(key=WANDB_KEY)
cb_autosave = ModelCheckpoint("ISPRS_semantic_labeling_Vaihingen_focal_gamma_2.keras",
                              mode="max",
                              save_best_only=True,
                              monitor="val_miou",
                              verbose=1)

cb_early_stop = EarlyStopping(patience=10,
                              verbose=1,
                              mode="auto",
                              monitor="val_miou")

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Semantic Segmentation",
    name="Semantic Segmantation - Experiment 2 - focal loss gamma 2",

    # track hyperparameters and run metadata
    config={
    "architecture": "CNN",
    "dataset": "ISPRS_semantic_labeling_Vaihingen",
    "bs": BATCH_SIZE
    }
)

cb_wandb = WandbMetricsLogger()

callbacks = [cb_autosave, cb_early_stop, cb_wandb]


### MODEL - U-Net Fit ###
unet.fit(data_gen_train,
          epochs=40,
          validation_data=data_gen_val,
          callbacks=callbacks
                    )
