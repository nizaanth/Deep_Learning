from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, \
                                    Flatten, \
                                    Dropout, \
                                    BatchNormalization, \
                                    Conv2D, \
                                    MaxPooling2D


def create_cnn(filters, 
               k, 
               input_shape, 
               n_classes):
    """Function to create a Convolutional neural network

    Args:
        filters (np.array): array with the number of filters
          used in each convolutional layer.
        k (int): kernel size used in each convolutional layer.
        input_shape (): shape of the input images [width,height,channels].
        n_classes (int): number of classes.

    Returns:
        Keras model: a Keras model created using the Sequential API.
    """

    model = Sequential()

    # Add convolutional layers
    for i, num_filters in enumerate(filters):
        if i == 0:
            # First convolutional layer needs input_shape
            model.add(Conv2D(num_filters, kernel_size=k, padding='same', activation='relu', input_shape=input_shape))
        else:
            model.add(Conv2D(num_filters, kernel_size=k, padding='same', activation='relu'))

        # Pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Batch normalization
        model.add(BatchNormalization())

    # Flatten layer to transition from convolutional to dense layers
    model.add(Flatten())

    # Fully connected (dense) layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(n_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model