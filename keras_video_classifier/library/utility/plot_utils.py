import matplotlib
matplotlib.use('Agg')
import numpy as np
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds,file_path='Confusion_Matrix_large.pdf'):
    plt.rcParams["figure.figsize"] = (50,50)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    ticks=np.linspace(0, 100,num=101)
    #print("Tick",ticks)
    #print("Tick_mark",tick_marks)
    plt.xticks(tick_marks, classes, fontsize=30, rotation=90) #old=50
    plt.yticks(tick_marks, classes, fontsize=30)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),fontsize=10,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black") #old = 12

    #plt.tight_layout()
    plt.grid(True)
    plt.ylabel('True label',fontsize=100)
    plt.xlabel('Predicted label',fontsize=100)
    plt.savefig(file_path)
    plt.savefig("CFS_test.jpg")
    
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
    """
    See: https://stackoverflow.com/a/26980472

    Identify most important features if given a vectorizer and binary classifier. Set n to the number
    of weighted features you would like to show. (Note: current implementation merely prints and does not
    return top classes.)
    """

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


def plot_history_2win(history, model_name, file_path):
    plt.subplot(211)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], color='g', label='Train')
    plt.plot(history.history['val_acc'], color='b', label='Validation')
    plt.legend(loc='best')

    plt.subplot(212)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='g', label='Train')
    plt.plot(history.history['val_loss'], color='b', label='Validation')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(file_path)


def create_history_plot(history, model_name, metrics=None):
    plt.title('Accuracy and Loss (' + model_name + ')')
    if metrics is None:
        metrics = {'acc', 'loss'}
    if 'acc' in metrics:
        plt.plot(history.history['acc'], color='g', label='Train Accuracy')
        plt.plot(history.history['val_acc'], color='b', label='Validation Accuracy')
    if 'loss' in metrics:
        plt.plot(history.history['loss'], color='r', label='Train Loss')
        plt.plot(history.history['val_loss'], color='m', label='Validation Loss')
    plt.legend(loc='best')

    plt.tight_layout()


def plot_history(history, model_name):
    create_history_plot(history, model_name)
    plt.show()


def plot_and_save_history(history, model_name, file_path, metrics=None):
    print("PASS plot_and_save_history")
    if metrics is None:
        metrics = {'acc', 'loss'}
    print("plot_and_save_history")
    create_history_plot(history, model_name, metrics)
    plt.savefig(file_path)
