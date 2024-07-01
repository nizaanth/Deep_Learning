from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, \
                                    Conv2D, \
                                    BatchNormalization, \
                                    ReLU, \
                                    AveragePooling2D, \
                                    UpSampling2D, \
                                    Concatenate                                    
                                    
from tensorflow.keras.applications.resnet50 import ResNet50, \
                                                   preprocess_input


def conv_bn_block(block_input,
                  num_filters=256,
                  kernel_size=3,
                  dilation_rate=1,
                  padding="same",
                  use_bias=False):
  """Function to apply a dilate convolution

  Args:
      block_input (keras.layer): input layer
      num_filters (int, optional): number of filters
        in the convolutional layer. Defaults to 256.
      kernel_size (int, optional): kernel size of each
        filter. Defaults to 3.
      dilation_rate (int, optional): dilation rate of the
        kernel. Defaults to 1.
      padding (str, optional): padding used for convolution.
        Defaults to "same".
      use_bias (bool, optional): If True, a bias is used
        for convolutions. Defaults to False.

  Returns:
      keras.layer: output layer
  """
  x = Conv2D(num_filters,
              kernel_size=kernel_size,
              dilation_rate=dilation_rate,
              padding=padding,
              use_bias=use_bias,
              kernel_initializer="he_normal",
              )(block_input)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return x


def DilatedSpatialPyramidPooling(dspp_input):
  """Function to create an Atrous Spatial Pyramid Pooling

  Args:
      dspp_input (keras.lyer): input layer

  Returns:
      keras.layer: output layer
  """
  # [16 x 16 x 256]
  dims = dspp_input.shape
  
  # Average Pooling: [16 x 16 x 256] => [1 x 1 x 256]
  x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
  # 256 Convolutions 1x1: [1 x 1 x 256] => [1 x 1 x 256]
  x = conv_bn_block(x, kernel_size=1, use_bias=True)
  # Up-sampling: [1 x 1 x 256] => [16 x 16 x 256]
  out_pool = UpSampling2D(size=(dims[-3] // x.shape[1], 
                                dims[-2] // x.shape[2]), 
                          interpolation="bilinear")(x)
  
  # 256 Atrous Convolutions, 1x1, rate=1: [16 x 16 x 256] => [16 x 16 x 256]
  out_1 = conv_bn_block(dspp_input, kernel_size=1, dilation_rate=1)
  # 256 Atrous Convolutions, 3x3, rate=6: [16 x 16 x 256] => [16 x 16 x 256]
  out_6 = conv_bn_block(dspp_input, kernel_size=3, dilation_rate=6)
  # 256 Atrous Convolutions, 3x3, rate=12: [16 x 16 x 256] => [16 x 16 x 256]
  out_12 = conv_bn_block(dspp_input, kernel_size=3, dilation_rate=12)
  # 256 Atrous Convolutions, 3x3, rate=18: [16 x 16 x 256] => [16 x 16 x 256]
  out_18 = conv_bn_block(dspp_input, kernel_size=3, dilation_rate=18)
  
  # Concatenate each output = [16 x 16 x 1280]
  x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])

  # 256 Convolutions, 1x1: [16 x 16 x 1280] => [16 x 16 x 256]
  output = conv_bn_block(x, kernel_size=1)

  return output


def get_deeplabv3plus(img_size, 
                      n_classes):
  """Function to create a DeepLabv3+ architecture

  Args:
      img_size (int): size of the input image
      n_classes (int): number of classes

  Returns:
      keras.Model: a keras model created using the
        functional API
  """
  model_input = Input(shape=(img_size, img_size, 3))
  # Pre-processing function
  x = preprocess_input(model_input)
  
  # ResNet50 backbone
  resnet50 = ResNet50(weights="imagenet", 
                      include_top=False, 
                      input_tensor=x)
  
  # Low-level features from backbone: [16 x 16 x 256]
  x = resnet50.get_layer("conv4_block6_2_relu").output
  # Apply ASPP - Atrous Spatial Pyramid Pooling: 
  # [16 x 16 x 256] => [16 x 16 x 256]
  x = DilatedSpatialPyramidPooling(x)
  # Up-sampling 4x: 
  # [16 x 16 x 256] => [64 x 64 x 256]
  input_a = UpSampling2D(size=(img_size // 4 // x.shape[1], 
                               img_size // 4 // x.shape[2]),
                         interpolation="bilinear")(x)
  
  # Low-level features from backbone: [64 x 64 x 64]
  input_b = resnet50.get_layer("conv2_block3_2_relu").output
  # 48 convolutions, 1x1: [64 x 64 x 48]
  input_b = conv_bn_block(input_b, num_filters=48, kernel_size=1)
  # Concatenate both outputs: [64 x 64 x 256] and [64 x 64 x 48]
  # results: [64 x 64 x 304]
  x = Concatenate(axis=-1)([input_a, input_b])

  # 256 convolutions, 3x3: [64 x 64 x 256]
  x = conv_bn_block(x)
  # 256 convolutions, 3x3: [64 x 64 x 256]
  x = conv_bn_block(x)
  # Up-sampling 4x: [256 x 256 x 256]
  x = UpSampling2D(size=(img_size // x.shape[1], 
                         img_size // x.shape[2]),
                   interpolation="bilinear")(x)
  
  # Final convolutions 1x1: [256 x 256 x #classes]
  model_output = Conv2D(n_classes, 
                        kernel_size=(1, 1), 
                        padding="same",
                        activation="softmax")(x)
  model = Model(inputs=model_input, 
                outputs=model_output)
  
  return model