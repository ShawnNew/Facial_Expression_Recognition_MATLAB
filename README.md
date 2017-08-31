# Facial_Expression_Recognition_MATLAB

This project is a graduate project from BUPT, and the purpose of the project is to design and implement a facial expression recongition system based on machein learning.

This project is built on Mac OS, MATLAB 2016a.

* The system is trained by JAFEE databse, which contains 213 expression pictures taken from Japanese Women. And the format of the pictures is tiff file.

* main.m file is the main file. The frame such as training the neural network, adjusting the parameters, predicting and calculating accuracy is contained in the main file.

* readImg.m file is aiming at gaining data from the picture library and store them in the workspace.

* displayData.m file is able to display the picture in the figure command.

* nnCostFunction.m file refers to the cost function of the neural network.

* randInitializeWeights.m file initialize the network randomly.

* computeNumericalGradient.m file compute the gradient of the neural network.

* checkNNGradients.m file can check the nerual network whether the network is convergent or not.

* predict.m file is used to predict the result given a certain set of samples.

* app.m file is the interface of the system in MATLAB.

