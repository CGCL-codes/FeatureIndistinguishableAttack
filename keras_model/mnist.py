from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Lambda
from keras.models import Sequential


def get_mnist_model(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='dense'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
