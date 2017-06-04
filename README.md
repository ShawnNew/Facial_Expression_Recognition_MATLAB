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









	·	程序运行操作说明：
	1.	如需对神经网络进行重新训练，请打开main.m文件，对于如下模块取消注释：Initialize the system、Extract the features、Define the training set and test set、Setup the parameters you will use for this exercise、Initializing Pameters、Training NN相关代码、Obtain Theta1 and Theta2 back from nn_params最后如果需要进行预测，则取消Implement Predict的注释。
	2.	如果要运行系统，则打开并运行app.m文件，选取网络参数result_0330_87.5%.mat网络参数，之后再打开JAFEE_Database文件夹中的任意一张照片，即可进行表情识别。
	3.	照相模块可以点按camera和shot进行测试。注：其中的人脸检测模块可以运行，但是识别准确率因为训练数据库不足的原因很低。
