# Introduction
### This repository contains some parts of code used in the paper: "ConvCoroNet: A deep convolutional neural network for Covid-19 detection using chest X-ray images". This code was executed in Google Colab taking advantage of the GPU access provided by Google.  ConvCoroNet is based on Inception v3 pre-trained model on ImageNet dataset and totally trained on a dataset prepared by collecting chest X-ray images of Covid-19, pneumonia and normal cases from publically available datasets . The experimental results of our proposed model show a high accuracy of 98.60%, sensitivity of 98,60% and specificity of 100% when detecting Covid-19 from chest X-ray images of Covid-19 positive cases, normal cases and pneumonia cases. Also shared links to download the data and models used during the training and testing phases are provided. 

# Used data
Referring to the article, the images used in this work have been downloaded from several sources. After the pre-processing operations (elimination of similar images, offline data augmentation), these images are available in Google Drive via the folder [covid-19_data](https://drive.google.com/drive/folders/1yfGRIyKvRRjcM6kcWfAMvB4kaq3wyQeB?usp=sharing). 

The notebook __preparing_datasets.ipynb__ allows to put the images data and their labels into nupmpy tables as mentioned in the following table: [data_train.npy](https://drive.google.com/file/d/1k-2eDTI0UZWvoFr-CmQ6VAYKGT0LZQhI/view?usp=sharing)  &&  [labels_train.npy](https://drive.google.com/file/d/1k-Uhkh2gYogsTiReG373aZRnQiqWok03/view?usp=sharing)

# Training and test
To evaluate the performance of ConCoroNet model, we used 5-fold cross-validation approach. The training set was randomly divided into 5 equal sets, Four of the 5 sets are used for model training, while the fifth is used for the validation set. This operation is repeated five times by shifting the training and the validation sets. Each time the performance of the model is reported. The notebook __ConvCoroNet_Training.ipynb__ allows to train our model for automatic detection of covid-19 from X-ray chest images. The models obtained for all folds: 

| fold #         | Validation Accuracy(%) | Downloads     |
|----------------|------------------------|---------------|
| fold 1          |        98,67           | [Download](https://drive.google.com/file/d/1-1eIXxQkIlnGNoUdbyBpJxsB3poE7nj0/view?usp=sharing) |
| fold 2    |        98,00           | [Download](https://drive.google.com/file/d/1-EKaLLUJQszgcgsKAqRnnNKbC5_A3c1g/view?usp=sharing) |
| fold 3          |        99,00           | [Download](https://drive.google.com/file/d/1-V2dWoS9E9e4ZCJZgQbbOP4Ro3pBisqj/view?usp=sharing) |
| fold 4       |        99,00           | [Download](https://drive.google.com/file/d/1-nPmhL_ZzsvxthrxE1D7gxPjPD1X7I7v/view?usp=sharing) |
| fold 5       |        98,33           | [Download](https://drive.google.com/file/d/102uSaE2s4C27E0n03AypBsafw-lr_tqo/view?usp=sharing) |
| Average       |        98,60           | --- |      
 
The notebook __ConvCoroNet_evaluation.ipynb__ displays results for accuracy, recall, precision and f1-score as well as confusion matrices and ROC curves for the 5-folds' models. It also calculates the average AUC obtained.
The __utils.py__ file containes some functions needed by the other files. 