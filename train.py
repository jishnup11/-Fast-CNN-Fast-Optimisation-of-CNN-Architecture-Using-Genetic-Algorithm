
from keras.datasets import mnist, cifar10
from keras.models import Sequential
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=3)

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    batch_size = 64
    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    classes = np.unique(y_train)
    nb_classes = len(classes)
   
    nRows,nCols,nDims = x_train.shape[1:]
    x_train = x_train.reshape(x_train.shape[0], nRows, nCols, nDims)
    x_test = x_test.reshape(x_test.shape[0], nRows, nCols, nDims)
    input_shape = (nRows, nCols, nDims)
    #print(input_shape)
    # Change to float datatype
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Scale the data to lie between 0 to 1
    x_train /= 255
    x_test /= 255

    # Change the labels from integer to categorical data
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

   

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)



def compile_model(network, nb_classes, input_shape):
    
    # Get our network parameters.
    nb_layers = network['nb_layers']
    activation = network['activation']
    learning_rate = network['learning_rate']
    weight_decay = network['weight_decay']
    momentum = network['momentum']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
        else:
            model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))  # hard-coded dropout

    # Output layer.
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes, activation='softmax'))
    sgd = SGD(lr=learning_rate, momentum=momentum, decay= weight_decay)

    model.compile(loss='categorical_crossentropy', optimizer= 'rmsprop',
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=30,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
