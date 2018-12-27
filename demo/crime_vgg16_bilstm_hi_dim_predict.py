import numpy as np
import sys
import os


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
    from keras_video_classifier.library.utility.crime.UCF_Crime_loader import load_ucf, scan_ucf_with_labels

    vgg16_include_top = False
    data_dir_path = os.path.join(os.path.dirname(__file__), 'very_large_data')
    model_dir_path = os.path.join(os.path.dirname(__file__), 'models','UCF-Anomaly-Detection-Dataset')

    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    load_ucf(data_dir_path)

    predictor = VGG16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)

    videos = scan_ucf_with_labels(data_dir_path, [label for (label, label_index) in predictor.labels.items()])
    
    print("PREDICT_VIDEOS", videos)
    
    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    
    print("video_file_path_list:", video_file_path_list)
    np.random.shuffle(video_file_path_list)

    correct_count = 0
    count = 0
    
    top5_correct_count=0
    
    print("Begin Predict Using UCF-CRIME-Model: ")
    print("Video_file_path_list: ", video_file_path_list)
    print(len(video_file_path_list))
    for video_file_path in video_file_path_list:
        label = videos[video_file_path]
        predicted_label = predictor.predict_top5(video_file_path)
        print('Top 1 predicted: ' + str(predicted_label[-1]) + ' actual: ' + label)
        correct_count = correct_count + 1 if label == predicted_label[-1] else correct_count
        count += 1
        accuracy = correct_count / count
        
        print('Top 5 predicted: ' + str(predicted_label) + ' actual: ' + label)
        top5_correct_count = top5_correct_count + 1 if  [s for s in predicted_label if label in s] else top5_correct_count
        accuracy_5 = top5_correct_count / count
        print('correct_count: ' + str(correct_count) + ' count: ' + str(count))
        print('top5_correct_count: ' + str(top5_correct_count) + ' count: ' + str(count))
        
        print('accuracy: ', accuracy)
        print('accuracy_top_5: ', accuracy_5)


if __name__ == '__main__':
    main()
