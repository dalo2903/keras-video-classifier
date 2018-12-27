import numpy as np
from keras import backend as K
import sys
import os

def main():
    K.set_image_dim_ordering('tf')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from keras_video_classifier.library.recurrent_networks import VGG16LSTMVideoClassifier
    from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels
    from keras_video_classifier.library.utility.plot_utils import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix

    vgg16_include_top = False
    data_dir_path = os.path.join(os.path.dirname(__file__), 'very_large_data')
    model_dir_path = os.path.join(os.path.dirname(__file__), 'models', 'UCF-101')
    config_file_path = VGG16LSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                     vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16LSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                     vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    load_ucf(data_dir_path)

    predictor = VGG16LSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)
    
    ### HARD_SPLIT = TRUE MEAN PREDICT IN UCF-101_Train folder ####
    videos = scan_ucf_with_labels(data_dir_path, [label for (label, label_index) in predictor.labels.items()],hard_split=True)
    #print("VIDEO KEY", videos)
    #print("VALUE",videos.values())
    all_label = videos.values()
    #print("TYPE", type(all_label))
    
    Total_label = []
    [Total_label.append(x) for x in all_label if x not in Total_label]
    
    print("Label: ",Total_label)
    #print("Label shape",np.shape(Total_label))
    with open("label.txt", "w") as output:
        output.write(str(Total_label))
    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    
    np.random.shuffle(video_file_path_list)

    correct_count = 0
    count = 0
    label_predicted = []
    label_true = []
    for video_file_path in video_file_path_list:
        label = videos[video_file_path]
        predicted_label = predictor.predict(video_file_path)
        
        label_predicted.append(predicted_label)
        label_true.append(label)
        
        print('predicted: ' + predicted_label + ' actual: ' + label)
        correct_count = correct_count + 1 if label == predicted_label else correct_count
        count += 1
        accuracy = correct_count / count
        print('correct_count: ' + str(correct_count) + ' count: ' + str(count))
        print('accuracy: ', accuracy)
    print("Label_Predicted",np.shape(label_predicted))
    print("True Label",np.shape(label_true))
    
    data_set_name = 'UCF-101'
    report_dir_path = os.path.join(os.path.dirname(__file__), 'reports', data_set_name)
    
    cm = confusion_matrix(label_true,label_predicted)    
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    import pandas as pd 
    df = pd.DataFrame(cm)
    df.to_csv("CM.csv")
    df = pd.DataFrame(recall)
    df.to_csv("ReCall.csv")
    df = pd.DataFrame(precision)
    df.to_csv("Precision.csv")
    np.set_printoptions(precision=2)
    
    plot_confusion_matrix(cm, Total_label,
                      title='Confusion matrix',normalize=True)
    
    #print(label_predicted)
    #print(label_true)

if __name__ == '__main__':
    main()
