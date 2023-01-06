import os
import librosa
import math
import json

dataset_path = "MusicData/genres_original"
json_path_1 = 'Data/CNN-LSTM_data.json'
json_path_2 = 'Data/KNN-SVM_data.json'
sample_rate = 22050
duration = 30
samples_per_track = sample_rate * duration


def save_mfcc(dataset_path, json_path, t, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(samples_per_track / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # loop through all the genres
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we're not at the root level
        if dirpath is not dataset_path:

            # save the semantic label
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
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

                    if t == 'CNN-LSTM':
                        # store mfcc for segment if it has expected length
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i - 1)
                            print("{}, segment: {}".format(file_path, s))

                    if t == 'KNN-SVM':
                        # store mean mfcc for segment if it has expected length
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc_mean.tolist())
                            data["labels"].append(i - 1)
                            print("{}, segment: {}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    t1 = 'CNN-LSTM'
    t2 = 'KNN-SVM'
    save_mfcc(dataset_path, json_path_1, t1, num_segments=10)
    save_mfcc(dataset_path, json_path_2, t2, num_segments=10)
