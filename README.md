This Repository contains all the work done in association with the MITx 6.86 Introduction to Machine Learning - From
Linear Models to Deep Learning as presented on the educational platform edx.org. I chose to complete this class after my first semester at TUM in order to get started with Machine Learning.

Since the class is not yet over, this repository will be extended by two further projects. The main focus of the
projects was a guided application of the course Material. So far the Repository contains following projects:

1. project0:
An introduction into the grading system used by the platform. There is no relevant course work covered here, however it
is included in the repository for completion purposes.

2. sentient_analysis:
This project was based on the learned material on Linear Classifiers. I implemented the perceptron and pegasos
algorithms for training linear classification models. These algorithms were then applied to a sentiment analysis of
written text reviews. Additionally, I implemented helper functions to help with the data extraction (bag of words
approach) and classification process. The implemented models were then tested and fine-tuned in the main.py file,
according to the course requirements.

3. mnist1:
This project explored different classification models for the MNIST digit dataset. Specifically, it focuses on non-linear features, linear regression, and kernel functions. In the linear_regeression.py file, I implemented a closed form linear regression for this task. In the file svm.py I implemented a Support Vector Machine that can classify the digits in a one vs. rest manner. This function is then used for the multiclass (0-9) classification. The file softmax.py features a softmax regression that is trained using the gradient descent algorithm. The file features.py explores Principal Component Analysis and dimensionality reduction, which aids the computation time of the previosly implemented models, as it reduces the datasize. The file kernel.py implements kernel functions for a more efficient computation using the Kernel Perceptron Algorithm. The file implements a polynomial kernel function, as well as a Gaussian RBF kernel function. All these discussed implementations were trained, fine-tuned, and tested/evaluated using the main.py file. 



4. mnist2:
This project focuses on the use of Neural Networks for the above mentioned MNIST classification task. The file neural_nets.py implemets a simple one layer feed forward neural network for arbitrary data point classification. This part was aimed at  implementing a NN from scratch in order to get more familiar with the underlying math. In the folder part2-mnist I explored different models for this classification task. The file nnet_fc.py features a fully connected NN, that I experimented on with different hyperparameters (batchsize, activation function, learning rate, momnetum). The aim of this task was to get an idea of how the hyperparameters affect model performance. The file nnet_cnn.py features my implementation of a Convolutional Neural Network for the classification task. The model makes use of two convolutional layers, one max pool layer and two fully connected linear layers. After local testing for 10 epochs, the model achieved an accuracy of +99%. Lastly, in the folderpart2-twodigit, I implemented 2 models that are able to classify the more complex overlapping two digit mnist dataset. In mlp.py I made use of another fully conected Feed Forward Neural Network with 2 different outputs. While simple to implement, this network has limited performance, as the digits are overlapping and not centered on the page anymore. To combat these issues, I implemented another CNN in the file conv.py. The inital convolutional and pooling layers are more efficient at picking up the features, which is why it reaches a higher accuracy. 
