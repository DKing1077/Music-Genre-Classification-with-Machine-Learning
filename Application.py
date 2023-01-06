from tensorflow.python.keras.models import load_model
import PySimpleGUI as sg
import librosa
import math
import json
import numpy as np
import threading
import pickle
import os


def main():

    font = 'Arial, 12'
    sg.theme('DarkBlue9')
    items = ['Convolutional Neural Network', 'Long Short Term Memory', 'K-Nearest Neighbour', 'Support Vector Machines']

    layout_column = [
        [sg.Text('Music Genre Classifier', font=font)],
        [sg.MLine('Music Genre Classification! \n 1) Choose a model \n 2) Upload a WAV audio sample \n 3) Run a classification',
            size=(80, 28), font='Tahoma 14', disabled=True, key='-MLine-', no_scrollbar=True)],
        [sg.Combo(items, size=(112, 1), readonly=True, key='-Listbox-')],
        [sg.Input(size=(82, 1)), sg.FileBrowse('Choose File', key='-Browse-', size=(26, 1))],
        [sg.Button('Run Classification', size=(100, 1))]
    ]

    layout = [[sg.Column(layout_column, element_justification='center')]]
    window = sg.Window('', layout, margins=(0, 0), finalize=True)

    # event loop
    while True:
        event, values = window.read(timeout=0)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        if event == 'Run Classification':
            # get interface values
            dataset_path = values['-Browse-']
            model_type = values['-Listbox-']
            if model_type == '':
                window['-MLine-'].print('\n')
                window['-MLine-'].print('Please select a model type!')
            elif dataset_path == '':
                window['-MLine-'].print('\n')
                window['-MLine-'].print('Please upload a WAV file!')
            else:
                # get file name
                name = os.path.basename(dataset_path)
                # display
                window['-MLine-'].update('Machine Learning Model : {}'.format(model_type))
                window['-MLine-'].print('\n')
                window['-MLine-'].print('File uploaded : {}'.format(name))
                window['-MLine-'].print('Processing samples ...')
                window.Refresh()
                # start threads
                threading.Thread(target=procesfile(dataset_path, model_type, num_segments=10), daemon=True).start()
                threading.Thread(target=make_prediction(window, model_type), daemon=True).start()


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    data = np.array(data["mfcc"])
    return data


def procesfile(dataset_path, model_type, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    file_path = dataset_path

    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    sample_rate = 22050
    duration = 30
    samples_per_track = sample_rate * duration

    num_samples_per_segment = int(samples_per_track / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # load audio file
    signal, sr = librosa.load(file_path, sr=sample_rate)

    # process segments extracting mfcc and storing data
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)

        mfcc = mfcc.T
        mfcc_mean = mfcc.mean(0)

        # store mfcc for segment if it has expected length CNN // LSTM
        if model_type in ['Convolutional Neural Network', 'Long Short Term Memory']:
            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())

        # store mean mfcc for segment if it has expected length KNN // SVM
        if model_type in ['K-Nearest Neighbour', 'Support Vector Machines']:
            if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc_mean.tolist())

    with open('Data/Input_data.json', "w") as fp:
        json.dump(data, fp, indent=4)


def make_prediction(window, model_type):
    window['-MLine-'].print('Loading models ...')
    window.Refresh()

    # load models
    if model_type == 'Convolutional Neural Network':
        model = load_model('Models/CNN_model_1.h5')

    elif model_type == 'Long Short Term Memory':
        model = load_model('Models/LSTM_model_1.h5')

    elif model_type == 'K-Nearest Neighbour':
        model = pickle.load(open('Models/KNN_model_1', 'rb'))

    elif model_type == 'Support Vector Machines':
        model = pickle.load(open('Models/SVM_model_1', 'rb'))

    # get predictions for 10 slices
    window['-MLine-'].print('Classifying please wait ...')
    window.Refresh()

    genres_array = []
    for i in range(9):

        window.Refresh()
        # load user processed input
        data = load_data('Data/input_data.json')
        # take data slice
        data = data[i]

        # amend input shape and make prediction
        if model_type == 'Convolutional Neural Network':
            data = data[np.newaxis, ...]
            data = data[..., np.newaxis]
            prediction = model.predict(data)
            predicted_index = np.argmax(prediction, axis=1)

        elif model_type == 'Long Short Term Memory':
            data = data[np.newaxis, ...]
            prediction = model.predict(data)
            predicted_index = np.argmax(prediction, axis=1)

        elif model_type == 'K-Nearest Neighbour':
            data = data[np.newaxis, ...]
            predicted_index = model.predict(data)

        elif model_type == 'Support Vector Machines':
            data = data[np.newaxis, ...]
            predicted_index = model.predict(data)

        # map predicted index to genre
        with open('Data/CNN-LSTM_data.json', "r") as fp:
            file = json.load(fp)
            mapping = np.array(file["mapping"])
            txt = mapping[predicted_index]
            print(predicted_index)
            genre = txt[0]
            genres_array.append(genre)

    print(genres_array)

    # get most common prediction
    classification = max(set(genres_array), key=genres_array.count)
    window['-MLine-'].print('\n')
    window['-MLine-'].print('   TrainingDataset: 1000 songs // 10 genres')

    if model_type == 'Convolutional Neural Network':
        window['-MLine-'].print('   Prediction Accuracy : 75%')
        window['-MLine-'].print('   Total params: 179,082')
        window['-MLine-'].print('   Trainable params: 178,698')
        window['-MLine-'].print('   Non-trainable params: 384')

    elif model_type == 'Long Short Term Memory':
        window['-MLine-'].print('   Prediction Accuracy : 73%')
        window['-MLine-'].print('   Total params: 127,690')
        window['-MLine-'].print('   Trainable params: 127,690')
        window['-MLine-'].print('   Non-trainable params: 0')

    elif model_type == 'K-Nearest Neighbour':
        window['-MLine-'].print('   Prediction Accuracy : 85%')

    elif model_type == 'Support Vector Machines':
        window['-MLine-'].print('   Prediction Accuracy : 54%')

    # display results
    window['-MLine-'].print('\n')
    window['-MLine-'].print('       Classification Complete!')
    window['-MLine-'].print('       The Genre Classification : {}'.format(classification).title())
    window['-MLine-'].print('\n')
    window.Refresh()


if __name__ == "__main__":
    main()
