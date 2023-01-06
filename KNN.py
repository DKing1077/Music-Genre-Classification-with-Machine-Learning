import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle

data_path = "Data/KNN-SVM_data.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y


def prepare_datasets(test_size):

    # load data
    x, y = load_data(data_path)

    # create train and test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":

    # get datasets
    x_train, x_test, y_train, y_test = prepare_datasets(0.25)

    # parameters
    grid_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # create and train model
    model = GridSearchCV(KNeighborsClassifier(n_neighbors=60), grid_params, n_jobs=-1)
    print('Training ...')
    model.fit(x_train, y_train)
    print('Training Finished')

    # save the model
    KNN = open('Models/KNN_model_1', 'wb')
    pickle.dump(model, KNN)

    # generate accuracy score and confusion matrix
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    print(accuracy)
    print('\n')
    print(confusion)

    # generate precision and recall
    precision = precision_score(y_test, predictions, pos_label='positive', average='micro')
    recall = recall_score(y_test, predictions, pos_label='positive', average='micro')
    print('\n')
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('\n')
