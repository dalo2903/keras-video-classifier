import urllib.request
import os
import sys
import patoolib

def scan_ucf(data_dir_path, limit):
    input_data_dir_path = data_dir_path + '/UCF-101'

    result = dict()

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                result[video_file_path] = f
        if dir_count == limit:
            break
    return result


def scan_ucf_with_labels(data_dir_path, labels):
    input_data_dir_path = data_dir_path + '/UCF-Anomaly-Detection-Dataset'

    result = dict()

    dir_count = 0
    for label in labels:
        file_path = input_data_dir_path + os.path.sep + label
        if not os.path.isfile(file_path):
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                result[video_file_path] = label
    return result



def load_ucf(data_dir_path):
    UFC_crime_data_dir_path = data_dir_path + "/UCF-Anomaly-Detection-Dataset"
    if not os.path.exists(UFC_crime_data_dir_path):
        print("Cannot find UCF_Crime dataset at:", data_dir_path)


def main():
    data_dir_path = '../very_large_data'
    load_ucf(data_dir_path)


if __name__ == '__main__':
    main()
