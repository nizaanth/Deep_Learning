from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, \
                                    MaxPooling2D, \
                                    Dropout, \
                                    Input
from tensorflow.keras.layers import Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam


def conv_block(x, 
               n_filters, 
               times=2):
  """Function to apply many convolutional layers

  Args:
      x (keras.layer): input layer
      n_filters (int): kernel size of the convolutional
        layers to be applied.
      times (int, optional): number of convolutional
        layers to be applied. Defaults to 2.

  Returns:
      keras.layer: output layer
  """
  for i in range(times):
    x = Conv2D(filters=n_filters,
              kernel_size=3,
              strides=1,
              padding="same",
              activation="relu",
              kernel_initializer="he_normal")(x)
  return x


def downsampling(x, 
                 n_filters, 
                 times=2):
  """Function with a downsampling block composed by a
    convolutional block, max pooling and dropout

  Args:
      x (keras.layer): input layer
      n_filters (int): kernel size of the convolutional
        layers to be applied
      times (int, optional): number of convolutional
        layers to be applied. Defaults to 2.

  Returns:
      keras.layer: output layer
  """
  feat = conv_block(x, n_filters, times=times)
  pool = MaxPooling2D(pool_size=2)(feat)
  pool = Dropout(rate=0.3)(pool)
  return feat,pool


def upsampling(input, 
               filters, 
               layer_concat=None):
  """Function with an upsampling block composed by a transpose
    convolution, concatenation, dropout and a convolutional block

  Args:
      input (keras.layer): input layer
      filters (int): kernel size of the convolutional layer and
        transpose convolutional layer
      layer_concat (keras.layer, optional): a layer with feature maps
        to be concatenated with the transpose convolution
          output. Defaults to None.

  Returns:
      keras.layer: output layer
  """
  x = Conv2DTranspose(filters=filters,
                      kernel_size=3,
                      strides=2,
                      padding="same")(input)
  if layer_concat is not None:
    x = concatenate([x, layer_concat])
  x = Dropout(rate=0.3)(x)
  x = conv_block(x, filters, times=2)
  return x


def get_unet(img_size, 
             n_classes):
  """Function to create a U-Net architecture

  Args:
      img_size (int): size of the input image
      n_classes (int): number of classes

  Returns:
      keras.Model: a keras model created using
        the functional API
  """
  # Input
  input = Input(shape=(img_size,img_size,3))
  # Downsampling
  f1,p1 = downsampling(input, 64, times=2)
  f2,p2 = downsampling(p1, 128, times=2)
  f3,p3 = downsampling(p2, 256, times=2)
  # Bottleneck
  blottleneck = conv_block(p3, 512, times=2)
  #Upsampling
  u7 = upsampling(blottleneck, 256, layer_concat=f3)
  u8 = upsampling(u7, 128, layer_concat=f2)
  u9 = upsampling(u8, 64, layer_concat=f1)
  # Output
  output = Conv2D(filters=n_classes,
                  kernel_size=1,
                  padding="same",
                  activation="softmax")(u9)
  model = Model(inputs=input, 
                outputs=output)

  return model