# Introduction
### This repository contains some parts of code used in the paper: "ConvCoroNet: A deep convolutional neural network optimized with iterative thresholding algorithm for Covid-19 detection using chest X-ray images". Our proposed model was implemented with Keras (Keras: The Python Deep Learning API, n.d.) on top of Tensorflow 2.0 framework (TensorFlow, n.d.). We used a machine equipped with a Ryzen 7 processor and a GPU RTX 3070 as accelerator. We experiment with five base models: InceptionV3, ResNet50V2, MobileNetV2, DenseNet121 and VGG16. ConvCoroNet is based on InceptionV3 pre-trained model on ImageNet dataset and totally retrained on a dataset of 3,000 images prepared by collecting chest X-ray images of Covid-19, pneumonia and normal cases from publically available datasets. We used as optimizer Itao (A new Iterative thresholding algorithm based optimizer for deep neural networks). The experimental results of our proposed model achives as overall metrics a high accuracy of 99.50%, a sensitivity of 99.50%, a specificity of 99.75%, a precision of 99.50% and f1-score of 99.50%. The results of other models using Itao and other state-of-the-art optimizers are also given. Shared links to download data and models used in this study are also provided. 

# Used data
Referring to the article, the images used in this work have been downloaded from several sources. These images are available in Google Drive via the folder [covid19_dataset](https://drive.google.com/drive/folders/1BB5PJEBiGzzRD240JPDjSAidXCNqJQpS?usp=sharing). 

The notebook __preparationOfDataset.ipynb__ allows to put the images data and their labels into nupmpy tables as mentioned in the following table: [new_3000_data_train_224.npy](https://drive.google.com/file/d/1i6SeB1VnW-OU8qQkMOt_Luvdn3kTu8TY/view?usp=sharing)  &&  [new_3000_labels_train_224.npy](https://drive.google.com/file/d/1vrXvhd2AVYO_5aUQ2t2q15-JHLrNG5nX/view?usp=sharing)

# Training and test
To evaluate the performance of ConCoroNet model, we used 5-fold cross-validation approach. The training set was randomly divided into 5 equal sets, Four of the 5 sets are used for model training, while the fifth is used for the validation set. This operation is repeated five times by shifting the training and the validation sets. Each time the performance of the model is reported. 
* The notebook color:green Covid-19_training_K-Fold_Itao_Inception.ipynb allows to train a model based on pre-trained InceptionV3 and retrained with Itao optimizer (Iterative Thresholding algorithm based optimizer). This model is the proposed model for this study.
* The notebook __Covid-19_training_K-Fold_Itao_ResNet_And_Others.ipynb__ allows to train a model based on other pre-trained models such as MobileNetV2, ResNet50V2, VGG16 and DenseNet121. Each of these models are retrained with Itao optimizer. The goal is to compare their performances with the proposed model.
* In the notebook __Covid-19_training_K-Fold_Inception_Other_Optimizers.ipynb__, we take the same proposed model and retrain it with other state-of-the-art optimizers like Adam, RMSprop and SGD to compare their results with those obtained with Itao.
* Notebook __Covid-19_training_InceptionV3_SVM_SGD_K-Fold.ipynb__ allows to train a model based on InceptionV3 as feature extractor and SVM (Support Vector Machine) as a classifier. The goal is to compare with proposed model.
* The notebook __Covid19_Evaluation_Models_Itao.ipynb__ allows to evaluate differents obtained models after training. It gives different metrics (recall, precision, f1-score ...), confusion matrices and ROC curves for the different used models trained with Itao.
* The notebook __Covid19_Evaluation_InceptionV3_Other_Optimizers.ipynb__ allows to evaluate the models based on InceptionV3 and trained with different optimizers.
* Notebook __Exploitation_matrices__ groups some functions that calculate results and plot barcharts used in paper.
* The __utils.py__ file containes some functions needed by the other files.      

The following table contains the 5 models obtained (5-fold cross-validation) after training the proposed model based on InceptionV3 with the new optimizer Itao: 

| fold #         | Validation Accuracy(%) | Downloads     |
|----------------|------------------------|---------------|
| fold 1          |        98,67           | [Download](https://drive.google.com/file/d/1-1eIXxQkIlnGNoUdbyBpJxsB3poE7nj0/view?usp=sharing) |
| fold 2    |        98,00           | [Download](https://drive.google.com/file/d/1-EKaLLUJQszgcgsKAqRnnNKbC5_A3c1g/view?usp=sharing) |
| fold 3          |        99,00           | [Download](https://drive.google.com/file/d/1-V2dWoS9E9e4ZCJZgQbbOP4Ro3pBisqj/view?usp=sharing) |
| fold 4       |        99,00           | [Download](https://drive.google.com/file/d/1-nPmhL_ZzsvxthrxE1D7gxPjPD1X7I7v/view?usp=sharing) |
| fold 5       |        98,33           | [Download](https://drive.google.com/file/d/102uSaE2s4C27E0n03AypBsafw-lr_tqo/view?usp=sharing) |
| Average       |        98,60           | --- |      
 
### The files exposed above permit the reproduction of the results obtained in the paper "ConvCoroNet: A deep convolutional neural network optimized with iterative thresholding algorithm for Covid-19 detection using chest X-ray images".
