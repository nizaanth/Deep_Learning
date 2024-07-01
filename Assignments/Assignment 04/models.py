from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, \
                                    Flatten, \
                                    Dropout, \
                                    BatchNormalization, \
                                    Conv2D, \
                                    MaxPooling2D
from tensorflow.keras.initializers import GlorotUniform, \
                                          HeNormal, \
                                          Ones, \
                                          Zeros, \
                                          RandomNormal


def create_cnn(filters, k, input_shape, n_classes, initializer):
    model = Sequential()
    
    for i, num_filters in enumerate(filters):
        if i == 0:
            model.add(Conv2D(num_filters, kernel_size=k, padding='same', activation='relu', kernel_initializer=initializer, input_shape=input_shape))
        else:
            model.add(Conv2D(num_filters, kernel_size=k, padding='same', activation='relu', kernel_initializer=initializer))
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax', kernel_initializer=initializer))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model