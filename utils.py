import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix

def plot_acc_loss(history, epochs):
    font = {
          'family':'Times New Roman',
          'size':12
          }

    matplotlib.rc('font', **font)
    acc = history['acc']
    loss = history['loss']
    val_acc = history['val_acc']
    val_loss = history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(1,epochs), acc[1:], label='Train_acc')
    plt.plot(range(1,epochs), val_acc[1:], label='Val_acc')
    plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(1,epochs), loss[1:], label='Train_loss')
    plt.plot(range(1,epochs), val_loss[1:], label='Val_loss')
    plt.title('Loss over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()     

def evaluation_metrics(model, data_val, labels_val, i, batch_size):
    # make predictions on the testing set
    print("[INFO] evaluating network...Fold: " + str(i))
    predIdxs = model.predict(data_val, batch_size=batch_size)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(labels_val.argmax(axis=1), predIdxs,
      target_names=['covid','normal','viral'])) #target_names=['covid','normal','viral']))
    class_names = ['Covid-19','Normal','Pneumonia'] #class_names = ['Covid-19','Normal','Pneumonia']
    #plotting learning curve and confusion matrix
    #help(model)
    from mlxtend.plotting import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
    font = {
          'family':'Times New Roman',
          'size':16
          }

    matplotlib.rc('font', **font)
    labels = np.argmax(labels_val, axis=1)
    mat = confusion_matrix(labels, predIdxs )
    fig, ax = plot_confusion_matrix(conf_mat=mat, figsize=(5,5),  show_absolute=True,  class_names=class_names, cmap='Purples' ) #PuRd 

def roc_curv_plot(labels_val, predIdxs, num_fold):
    font = {
      'family':'Times New Roman',
      'size':13
      }

    matplotlib.rc('font', **font)
    #plt.figure()
    class_names = ['Covid-19','Normal','Pneumonia'] #['Covid-19','Normal','Pneumonia']
    n_classes = 3 #3
    # Plot linewidth.
    lw = 3
  
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_val[:, i], predIdxs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
  
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_val.ravel(), predIdxs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  
    # Compute macro-average ROC curve and ROC area
  
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
  
    # Finally average it and compute AUC
    mean_tpr /= n_classes
  
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
  
    # Plot all ROC curves
    plt.figure(1, figsize=(8,8))
    #plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.4f})'
    #              ''.format(roc_auc["micro"]),
    #         color='navy', linestyle=':', linewidth=4)
  
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (auc = {0:0.5f})'
                  ''.format(roc_auc["macro"]),
            color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['darkorange', 'green', 'cornflowerblue']) #, 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} vs rest of classes (auc = {1:0.5f})' #vs rest of classes
                ''.format(class_names[i], roc_auc[i]))
  
    #plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.4f})'
    #             ''.format(class_names[0], roc_auc[0]))
  
    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve for ConvCoroNet (Fold " + str(num_fold)+ ")") 
    plt.legend(loc="lower right")
    plt.show()
    
    # Zoom in view of the upper left corner.
    plt.figure(2,figsize=(8,8))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1.01)
    #plt.plot(fpr["micro"], tpr["macro"],
    #         label='micro-average ROC curve (area = {0:0.4f})'
    #               ''.format(roc_auc["micro"]),
    #         color='navy', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (auc = {0:0.5f})'
                   ''.format(roc_auc["macro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    colors = cycle(['darkorange', 'green', 'cornflowerblue']) #, 'cornflowerblue']) #cornflowerblue
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} vs rest of classes (auc = {1:0.5f})' #vs rest of classes
                 ''.format(class_names[i], roc_auc[i]))
    #plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw,
    #         label='ROC curve of class {0} (area = {1:0.4f})'
    #         ''.format(class_names[0], roc_auc[0]))


    #plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve for ConvCoroNet (Fold " + str(num_fold)+ ") upper_left corner details")
    plt.legend(loc="lower right")
    plt.show()
        
    
