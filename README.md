# Facial_Expression_Recognition_MATLAB

This project is a graduate project from BUPT, and the purpose of the project is to design and implement a facial expression recongition system based on machein learning.

This project is built on Mac OS, MATLAB 2016a.

<li></li>




基于机器学习的人脸表情识别系统的运行说明书
	·	该人脸表情识别程序所需的运行环境是MATLAB，需将文件夹fer的路径加到MATLAB的当前路径中以保证程序的正常运行。
	·	该程序是在Mac OS10.12.3 下的2016a版本的MATLAB下实现。
	·	源代码介绍
	1.	JAFEE_Database文件夹是工程所需的数据库，其中包含213张表情图片，格式为256*256的tiff文件。
	2.	主文件为main.m文件，其中是程序的主要框架，包括了神经网络的训练、参数调整、模型预测以及准确率的对比。
	3.	readImg.m文件，是程序从JAFEE_Database中提取数据到运行空间的文件，文件中包括了系统的特征提取部分。
	4.	displayData.m，是现实图片部分。
	5.	nnCostFunction.m，是神经网络的误差函数的文件。
	6.	randInitializeWeights.m，是随机初始化网络的文件。
	7.	computeNumericalGradient.m，是计算神经网络梯度的文件。
	8.	checkNNGradients.m, 是检查神经网络是否收敛的文件。
	9.	app.m，是系统的图形界面。
	10.	predict.m，是模型预测的文件。
	11.	result_0330_87.5%.mat，是训练好的准确率为87.5%的网络参数。
	·	程序运行操作说明：
	1.	如需对神经网络进行重新训练，请打开main.m文件，对于如下模块取消注释：Initialize the system、Extract the features、Define the training set and test set、Setup the parameters you will use for this exercise、Initializing Pameters、Training NN相关代码、Obtain Theta1 and Theta2 back from nn_params最后如果需要进行预测，则取消Implement Predict的注释。
	2.	如果要运行系统，则打开并运行app.m文件，选取网络参数result_0330_87.5%.mat网络参数，之后再打开JAFEE_Database文件夹中的任意一张照片，即可进行表情识别。
	3.	照相模块可以点按camera和shot进行测试。注：其中的人脸检测模块可以运行，但是识别准确率因为训练数据库不足的原因很低。
