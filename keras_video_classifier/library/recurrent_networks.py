from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD,RMSprop
import os
import numpy as np

from keras_video_classifier.library.utility.frame_extractors.vgg16_feature_extractor import extract_vgg16_features_live, \
    scan_and_extract_vgg16_features
from keras_video_classifier.library.utility.frame_extractors.resnet_feature_extractor import extract_resnet_features_live, \
    scan_and_extract_resnet_features

BATCH_SIZE = 1
NUM_EPOCHS = 50
VERBOSE = 1 # Progress bar mode
HIDDEN_UNITS = 256
#MAX_ALLOWED_FRAMES = 200
#EMBEDDING_SIZE = 100

K.set_image_dim_ordering('tf')


def generate_batch(x_samples, y_samples):
    num_batches = len(x_samples) // BATCH_SIZE

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            yield np.array(x_samples[start:end]), y_samples[start:end]


class VGG16BidirectionalLSTMVideoClassifier(object):
    model_name = 'vgg16-bidirectional-lstm'

    def __init__(self):
        self.num_input_tokens = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.vgg16_model = None
        self.expected_frames = None
        self.vgg16_include_top = True
        self.config = None

    def create_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=HIDDEN_UNITS, return_sequences=True),
                                input_shape=(self.expected_frames, self.num_input_tokens)))
        model.add(Bidirectional(LSTM(HIDDEN_UNITS)))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.nb_classes))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model

    @staticmethod
    def get_config_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-config.npy'
        else:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-weights.h5'
        else:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-architecture.json'
        else:
            return model_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-architecture.json'

    def load_model(self, config_file_path, weight_file_path):
        if os.path.exists(config_file_path):
            print('loading configuration from ', config_file_path)
        else:
            raise ValueError('cannot locate config file {}'.format(config_file_path))

        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.vgg16_include_top = config['vgg16_include_top']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])
        self.config = config

        self.model = self.create_model()
        if os.path.exists(weight_file_path):
            print('loading network weights from ', weight_file_path)
        else:
            raise ValueError('cannot local weight file {}'.format(weight_file_path))

        self.model.load_weights(weight_file_path)

        print('build vgg16 with pre-trained model')
        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

    def predict(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_idx2word[predicted_class]
        return predicted_label
    
    def labels_idx2word_top5(predicted_classes):
        predict_labels = []
        for i,predicted_class in predicted_classes:
            label = self.labels_idx2word[predicted_class]
            predict_labels.append(label)
        return predict_labels
    
    def predict_top5(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predict = self.model.predict(np.array([x]))
        print(predict)
        predicted_class = np.argsort(predict[0])[-5:]
        print(predicted_class)
        predict_labels = []
        #top_values= [class_prob[i] for i in np.argsort(class_prob)[-5:]]
        for _class in predicted_class:
            #print(_class)
            label = self.labels_idx2word[_class]
            predict_labels.append(label)
            
        #predicted_label = labels_idx2word_top5(predicted_class)
        #predicted_class = np.argmax(self.model.predict(np.array([x]))[:5])
        #predicted_label = self.labels_idx2word[predicted_class]
        
        print(predict_labels)
        
        return predict_labels
        #return predicted_label
    
    
    
    def fit(self, data_dir_path, model_dir_path, vgg16_include_top=True, data_set_name='UCF-101', test_size=0.2,
            random_state=42):

        self.vgg16_include_top = vgg16_include_top

        config_file_path = self.get_config_file_path(model_dir_path, vgg16_include_top)
        weight_file_path = self.get_weight_file_path(model_dir_path, vgg16_include_top)
        architecture_file_path = self.get_architecture_file_path(model_dir_path, vgg16_include_top)

        self.vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        self.vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        feature_dir_name = data_set_name + '-VGG16-Features'
        if not vgg16_include_top:
            feature_dir_name = data_set_name + '-VGG16-HiDimFeatures'
        max_frames = 0
        self.labels = dict()
        x_samples, y_samples = scan_and_extract_vgg16_features(data_dir_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=self.vgg16_model,
                                                               data_set_name=data_set_name)
        self.num_input_tokens = x_samples[0].shape[1]
        frames_list = []
        for x in x_samples:
            frames = x.shape[0]
            frames_list.append(frames)
            max_frames = max(frames, max_frames)
        self.expected_frames = int(np.mean(frames_list))
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[0]
            if frames > self.expected_frames:
                x = x[0:self.expected_frames, :]
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
                temp[0:frames, :] = x
                x_samples[i] = temp
        for y in y_samples:
            if y not in self.labels:
                self.labels[y] = len(self.labels)
        print(self.labels)
        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)

        y_samples = np_utils.to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = self.labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames
        config['vgg16_include_top'] = self.vgg16_include_top

        self.config = config

        np.save(config_file_path, config)

        model = self.create_model()
        open(architecture_file_path, 'w').write(model.to_json())

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=random_state)

        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)

        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE

        checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
        history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                      epochs=NUM_EPOCHS,
                                      verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                      callbacks=[checkpoint])
        model.save_weights(weight_file_path)

        return history


class VGG16LSTMVideoClassifier(object):
    model_name = 'vgg16-lstm'
    
    def __init__(self):
        self.num_input_tokens = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.vgg16_model = None
        self.expected_frames = None
        self.vgg16_include_top = None
        self.config = None
        
    @staticmethod
    def get_config_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-config.npy'
        else:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-hi-dim-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-weights.h5'
        else:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-hi-dim-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path, vgg16_include_top=None):
        if vgg16_include_top is None:
            vgg16_include_top = True
        if vgg16_include_top:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-architecture.json'
        else:
            return model_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-hi-dim-architecture.json'

    def create_model(self):
        model = Sequential()
        model.add(
            LSTM(units=HIDDEN_UNITS, input_shape=(None, self.num_input_tokens), return_sequences=False, dropout=0.5))
        
        #model.add(
        #    LSTM(units=HIDDEN_UNITS, input_shape=(None, self.num_input_tokens), return_sequences=True, dropout=0.5))
        #model.add(
        #    LSTM(units=HIDDEN_UNITS, input_shape=(None, HIDDEN_UNITS), return_sequences=True, dropout=0.4))
        #model.add(
        #    LSTM(units=HIDDEN_UNITS, input_shape=(None, HIDDEN_UNITS), return_sequences=False, dropout=0.3))
        
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])      
        for layer in model.layers:
            print("Input shape:" , layer.input_shape)
        model.summary()
        return model

    def load_model(self, config_file_path, weight_file_path):

        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.vgg16_include_top = config['vgg16_include_top']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])

        self.model = self.create_model()
        self.model.load_weights(weight_file_path)

        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

    def predict(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_idx2word[predicted_class]
        return predicted_label
    
    def predict_top5(self, video_file_path):
        x = extract_vgg16_features_live(self.vgg16_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[:5])
        predicted_label = self.labels_idx2word[predicted_class]
        print(predicted_label)
        return predicted_label
    
    def fit(self, data_dir_path, model_dir_path, vgg16_include_top=True, data_set_name='UCF-101', test_size=0.2, random_state=42):
        self.vgg16_include_top = vgg16_include_top

        config_file_path = self.get_config_file_path(model_dir_path, vgg16_include_top)
        weight_file_path = self.get_weight_file_path(model_dir_path, vgg16_include_top)
        architecture_file_path = self.get_architecture_file_path(model_dir_path, vgg16_include_top)

        vgg16_model = VGG16(include_top=self.vgg16_include_top, weights='imagenet')
        vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.vgg16_model = vgg16_model

        feature_dir_name = data_set_name + '-VGG16-Features'
        if not vgg16_include_top:
            feature_dir_name = data_set_name + '-VGG16-HiDimFeatures'
        max_frames = 0
        self.labels = dict()
        x_samples, y_samples = scan_and_extract_vgg16_features(data_dir_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=self.vgg16_model,
                                                               data_set_name=data_set_name)
        self.num_input_tokens = x_samples[0].shape[1]
        print("num_input_tokens= :", self.num_input_tokens) # = 7x7x512
        frames_list = []
        for x in x_samples:
            frames = x.shape[0]
            #print("frames in x: = " , frames)
            
            frames_list.append(frames)
            
            #print("frames_list: = " , frames_list)
            
            max_frames = max(frames, max_frames)
            
            #print("max_frames: = " , max_frames)   
            
            self.expected_frames = int(np.mean(frames_list))
            
            #print("expected_frames: = " , self.expected_frames)
            
        print("frames_list: = " , frames_list)
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        print("len(x_samples): = ", len(x_samples))
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[0]
            #print("x.shape",x.shape) #(frames,7x7x512)
            if frames > self.expected_frames:
                x = x[0:self.expected_frames, :]
                #print("x.shape if frame > expected frame ",x.shape)
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
                temp[0:frames, :] = x
                x_samples[i] = temp
                #print("x.shape if frame < expected frame (temp) ",temp.shape)
        print("WERAWERAWER")
        for y in y_samples:
            print("for y in y_samples: passed ")
            if y not in self.labels:
                self.labels[y] = len(self.labels)
        print("self.labels:" , self.labels)
        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)

        y_samples = np_utils.to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = self.labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames
        config['vgg16_include_top'] = self.vgg16_include_top
        self.config = config

        np.save(config_file_path, config)

        
        print("PASS NP SAVE in RCNN")
        
        model = self.create_model()
        open(architecture_file_path, 'w').write(model.to_json())
        print("PASS model = self.create_model() in RCNN")

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=random_state)
        print("Xtrain", np.shape(Xtrain))
        print("Xtest", np.shape(Xtest))
        print("Ytrain", np.shape(Ytrain))
        print("Ytest", np.shape(Ytest))
        
        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)
        
        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE

        checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
        history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                      epochs=NUM_EPOCHS,
                                      verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                      callbacks=[checkpoint])
        model.save_weights(weight_file_path)

        return history
    
class ResnetLSTMVideoClassifier(object):
    model_name = 'resnet-lstm'
    
    def __init__(self):
        self.num_input_tokens = None
        self.nb_classes = None
        self.labels = None
        self.labels_idx2word = None
        self.model = None
        self.resnet_model = None
        self.expected_frames = None
        self.resnset_include_top = None
        self.config = None

    @staticmethod
    def get_config_file_path(model_dir_path, resnet_include_top=None):
        if resnet_include_top is None:
            resnet_include_top = True
        if resnet_include_top:
            return model_dir_path + '/' + ResnetLSTMVideoClassifier.model_name + '-config.npy'
        else:
            return model_dir_path + '/' + ResnetLSTMVideoClassifier.model_name + '-hi-dim-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path, resnet_include_top=None):
        if resnet_include_top is None:
            resnet_include_top = True
        if resnet_include_top:
            return model_dir_path + '/' + ResnetLSTMVideoClassifier.model_name + '-weights.h5'
        else:
            return model_dir_path + '/' + ResnetLSTMVideoClassifier.model_name + '-hi-dim-weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path, resnet_include_top=None):
        if resnet_include_top is None:
            resnet_include_top = True
        if resnet_include_top:
            return model_dir_path + '/' + ResnetLSTMVideoClassifier.model_name + '-architecture.json'
        else:
            return model_dir_path + '/' + ResnetLSTMVideoClassifier.model_name + '-hi-dim-architecture.json'

    def create_model(self):
        model = Sequential()
        model.add(
            LSTM(units=HIDDEN_UNITS, input_shape=(None, self.num_input_tokens), return_sequences=False, dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])      
        return model

    def load_model(self, config_file_path, weight_file_path):

        config = np.load(config_file_path).item()
        self.num_input_tokens = config['num_input_tokens']
        self.nb_classes = config['nb_classes']
        self.labels = config['labels']
        self.expected_frames = config['expected_frames']
        self.resnet_include_top = config['resnet_include_top']
        self.labels_idx2word = dict([(idx, word) for word, idx in self.labels.items()])

        self.model = self.create_model()
        self.model.load_weights(weight_file_path)

        resnet_model = ResNet50(include_top=self.resnet_include_top, weights='imagenet')
        resnet_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.resnet_model = resnet_model

    def predict(self, video_file_path):
        x = extract_resnet_features_live(self.resnet_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[0])
        predicted_label = self.labels_idx2word[predicted_class]
        return predicted_label
    
    def predict_top5(self, video_file_path):
        x = extract_resnet_features_live(self.resnet_model, video_file_path)
        frames = x.shape[0]
        if frames > self.expected_frames:
            x = x[0:self.expected_frames, :]
        elif frames < self.expected_frames:
            temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
            temp[0:frames, :] = x
            x = temp
        predicted_class = np.argmax(self.model.predict(np.array([x]))[:5])
        predicted_label = self.labels_idx2word[predicted_class]
        print(predicted_label)
        return predicted_label
    
    def fit(self, data_dir_path, model_dir_path, resnet_include_top=True, data_set_name='UCF-101', test_size=0.2, random_state=42):
        self.resnet_include_top = resnet_include_top

        config_file_path = self.get_config_file_path(model_dir_path, resnet_include_top)
        weight_file_path = self.get_weight_file_path(model_dir_path, resnet_include_top)
        architecture_file_path = self.get_architecture_file_path(model_dir_path, resnet_include_top)

        resnet_model = ResNet50(include_top=self.resnet_include_top, weights='imagenet')
        resnet_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.resnet_model = resnet_model

        feature_dir_name = data_set_name + '-Resnet-Features'
        if not resnet_include_top:
            feature_dir_name = data_set_name + '-Resnet-HiDimFeatures'
        max_frames = 0
        self.labels = dict()
        x_samples, y_samples = scan_and_extract_resnet_features(data_dir_path,
                                                               output_dir_path=feature_dir_name,
                                                               model=self.resnet_model,
                                                               data_set_name=data_set_name)
        self.num_input_tokens = x_samples[0].shape[1]
        print("num_input_tokens= :", self.num_input_tokens) # = 7x7x512
        frames_list = []
        for x in x_samples:
            frames = x.shape[0]
            print("frames in x: = " , frames)
            
            frames_list.append(frames)
            
            #print("frames_list: = " , frames_list)
            
            max_frames = max(frames, max_frames)
            
            print("max_frames: = " , max_frames)   
            
            self.expected_frames = int(np.mean(frames_list))
            
            print("expected_frames: = " , self.expected_frames)
            
        print("frames_list: = " , frames_list)
        print('max frames: ', max_frames)
        print('expected frames: ', self.expected_frames)
        print("len(x_samples): = ", len(x_samples))
        for i in range(len(x_samples)):
            x = x_samples[i]
            frames = x.shape[0]
            #print("x.shape",x.shape) #(frames,7x7x512)
            if frames > self.expected_frames:
                x = x[0:self.expected_frames, :]
                #print("x.shape if frame > expected frame ",x.shape)
                x_samples[i] = x
            elif frames < self.expected_frames:
                temp = np.zeros(shape=(self.expected_frames, x.shape[1]))
                temp[0:frames, :] = x
                x_samples[i] = temp
                #print("x.shape if frame < expected frame (temp) ",temp.shape)
        for y in y_samples:
            if y not in self.labels:
                self.labels[y] = len(self.labels)
        print("self.labels:" , self.labels)
        for i in range(len(y_samples)):
            y_samples[i] = self.labels[y_samples[i]]

        self.nb_classes = len(self.labels)

        y_samples = np_utils.to_categorical(y_samples, self.nb_classes)

        config = dict()
        config['labels'] = self.labels
        config['nb_classes'] = self.nb_classes
        config['num_input_tokens'] = self.num_input_tokens
        config['expected_frames'] = self.expected_frames
        config['resnet_include_top'] = self.resnet_include_top
        self.config = config

        np.save(config_file_path, config)

        model = self.create_model()
        open(architecture_file_path, 'w').write(model.to_json())

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_samples, y_samples, test_size=test_size,
                                                        random_state=random_state)
        #print("Xtrain", np.shape(Xtrain))
        #print("Xtest", np.shape(Xtest))
        #print("Ytrain", np.shape(Ytrain))
        #print("Ytest", np.shape(Ytest))
        
        train_gen = generate_batch(Xtrain, Ytrain)
        test_gen = generate_batch(Xtest, Ytest)
        
        train_num_batches = len(Xtrain) // BATCH_SIZE
        test_num_batches = len(Xtest) // BATCH_SIZE

        checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=True)
        history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                      epochs=NUM_EPOCHS,
                                      verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                      callbacks=[checkpoint])
        model.save_weights(weight_file_path)

        return history
