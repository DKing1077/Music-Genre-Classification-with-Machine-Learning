import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.python.keras as keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_path = "Data/CNN-LSTM_data.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return x, y


def prepare_datasets(test_size, validation_size):

    # load data
    x, y = load_data(data_path)

    # create train, validation and test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    # add an axis to input sets
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_test, x_validation, y_train, y_test, y_validation


def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def build_model(input_shape):

    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":

    # get train, validation, test splits
    x_train, x_test, x_validation, y_train, y_test, y_validation = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_model(input_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train model
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=60)
    plot_history(history)

    # save the model
    model.save('Models/CNN_model_1.h5')

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print(model.summary())

    # generate model summary
    model.summary()
    print('\n')

    # generate confusion matrix
    saved_model = load_model('Models/CNN_model_1.h5')
    predictions = saved_model.predict(x_test)
    prediction_indexes = np.argmax(predictions, axis=1)
    confusion = tf.math.confusion_matrix(labels=y_test, predictions=prediction_indexes)
    print(confusion)

    # generate precision and recall
    precision = precision_score(y_test, prediction_indexes, pos_label='positive', average='micro')
    recall = recall_score(y_test, prediction_indexes, pos_label='positive', average='micro')
    print('\n')
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('\n')
