import cv2
import os
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD

MAX_NB_CLASSES = 101


def extract_resnet_features_live(model, video_input_file_path):
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()            
            features.append(feature)
            count = count + 0.5
    unscaled_features = np.array(features)
    return unscaled_features


def extract_resnet_features(model, video_input_file_path, feature_output_file_path):
    # if feature_output exist, load it and didn't use VGG16
    
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        #print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel() #Flatten nd.array to [1,2,3,4,5...]
            features.append(feature)
            #print("FEATURE.shape= :" , np.shape(feature)) # features shape = flatten(7x7x512) of VGG16
            count = count + 0.5
            #print("count: = " ,count)
    unscaled_features = np.array(features)
    # print("unscaled_features.shape= :" , np.shape(unscaled_features)) # shape = (number_of_frame,7x7x512)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def scan_and_extract_resnet_features(data_dir_path, output_dir_path, model=None, data_set_name=None):
    if data_set_name is None:
        data_set_name = 'UCF-101'

    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + output_dir_path

    if model is None:
        model = ResNet50(include_top=True, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                x = extract_resnet_features(model, video_file_path, output_feature_file_path) # x.shape=(number_of_frames,7x7x512)
                #print("X.shape ========: ", np.shape(x))
                #print("F ======== : ",f)
                y = f # f = label of folder
                # print("X====: ",x) # features extraction of VGG16 as np array.
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples

