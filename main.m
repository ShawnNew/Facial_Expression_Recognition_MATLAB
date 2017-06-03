%% Facial Expression Recognition - Using Neural Network Learning Algorithm

%  Instructions
%  ------------
%  This file extract the image data form JAFEE_Databse, and extract the
%  feature using the method of LBP. Then put the data into the workplace
%  called 'imgdata.mat', y contains the label of the correspond
%  image. In this machine learning system, Neural Network is used to train
%  the parameters all_theta.
%

%% ====================== Initialize the system ===========================
clear ; close all; clc

%% ======================= Loading and Visualizing Data ===================
%  Loading and visualizing the dataset. 
%  You will be working with a dataset that contains facial expression.
%

% Load Training Data
fprintf('Loading the data for visualizing ...\n')
load('optDataset.mat');
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:4);
displayData(X(sel, :));


%% ==================== Extract the features =====================
clear;
fprintf('Loading the raw image data from "imgdata.mat"...\n')
load('imgdata.mat');
% Compare different methods to extract featrues, use LBP only, 2DPCA only,
% DCT only, 2DPCA plus LBP, 2DPCA plus DCT, LBP plus DCT and DCT plus LBP
% plus 2DPCA. Invoke different functions to modify the features.
m = size(X,1);

% Extract the features using LBP only
f_LBP = zeros(m, 256);
for i = 1:m
   f_LBP(i, :) = lbp(reshape(X(i,:), 256, 256)); 
end

% Extract the features using 2DPCA only
img = reshape(X, m, 256, 256);
f_2DPCA = pca_2d(img, 5);

% Extract the features using DCT only
f_DCT = dct(img, 1000);    % use the first 1000 features of DCT

% Extract the features using 2DPCA plus DCT
f_D_2DPCA = dct(f_2DPCA, 100);

% Extract the features using LBP plus DCT
f_L_DCT = dct(f_LBP, 100);



%% ===============  Define the training set and test set  =================
% Set the data into three divisions: training set, test set, validation set
% Write 



% clear;
% fprintf('Loading the data for training ...\n')
% % load('Dataset_0_0.5.mat');
% load('lbpDataset.mat');


m = size(X, 1);
% Define the ratio of each data group(training, validation, test)
trainRatio = 0.7;  
valRatio = 0.15;
testRatio = 0.15;
sel = randperm(m);
sel_train = sel(1: floor(length(sel) * trainRatio));
sel_val = sel(ceil(length(sel) * trainRatio) :...
    floor(length(sel) * (trainRatio + valRatio)));
sel_test = sel(ceil(length(sel) * (trainRatio + valRatio)): end);
% Divide the data group
% X_train = X(sel_train, :);
y_train = y(sel_train, :);
% X_val = X(sel_val, :);
y_val = y(sel_val, :);
% X_test = X(sel_test, :);
y_test = y(sel_test, :);

% pca features
n_pca = size(f_2DPCA, 2) * size(f_2DPCA, 3);
pca_train = reshape(f_2DPCA, m, n_pca);
pca_train = pca_train(sel_train, :);
pca_val = reshape(f_2DPCA, m, n_pca);
pca_val = pca_val(sel_val, :);
pca_test = reshape(f_2DPCA, m, n_pca);
pca_test = pca_test(sel_test, :);

% dct features
n_dct = size(f_DCT, 2);
dct_train = reshape(f_DCT, m, n_dct);
dct_train = dct_train(sel_train, :);
dct_val = reshape(f_DCT, m, n_dct);
dct_val = dct_val(sel_train, :);
dct_test = reshape(f_DCT, m, n_dct);
dct_test = dct_test(sel_train, :);

% dct and pca features
n_d_pca = size(f_D_2DPCA, 2);
d_pca_train = reshape(f_D_2DPCA,m,n_d_pca);
d_pca_train = d_pca_train(sel_train, :);
d_pca_val = reshape(f_D_2DPCA,m,n_d_pca);
d_pca_val = d_pca_val(sel_train, :);
d_pca_test = reshape(f_D_2DPCA,m,n_d_pca);
d_pca_test = d_pca_test(sel_train, :);

% lbp features
n_lbp = size(f_LBP,2);
lbp_train = reshape(f_LBP,m,n_lbp);
lbp_train = lbp_train(sel_train, :);
lbp_val = reshape(f_LBP,m,n_lbp);
lbp_val = lbp_val(sel_train, :);
lbp_test = reshape(f_LBP,m,n_lbp);
lbp_test = lbp_test(sel_train, :);

% dct and lbp features
n_l_dct = size(f_L_DCT, 2);
l_dct_train = reshape(f_L_DCT,m,n_l_dct);
l_dct_train = l_dct_train(sel_train, :);
l_dct_val = reshape(f_L_DCT,m,n_l_dct);
l_dct_val = l_dct_val(sel_train, :);
l_dct_test = reshape(f_L_DCT,m,n_l_dct);
l_dct_test = l_dct_test(sel_train, :);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Setup the parameters you will use for this exercise ========
input_layer_size  = size(X, 2);  % Number of elements of each colomn of X
hidden_layer_size = 10;          % Manually defined
num_labels = size(y, 2);         % 7 labels, from 1 to 7


% setup parameters for pca features
pca_input_layer_size = size(pca_train, 2);

% setup parameters for dct features
dct_input_layer_size = size(dct_train, 2);
% setup parameters for dct and pca featrues
d_pca_input_layer_size = size(d_pca_train, 2);
% setup parameters for lbp features
lbp_input_layer_size = size(lbp_train, 2);
% setup parameters for lbp and dct features
l_dct_input_layer_size = size(l_dct_train, 2);

%% ========================= Initializing Pameters ========================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')



initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);



% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



% initialize pca para
pca_initial_Theta1 = randInitializeWeights(pca_input_layer_size, hidden_layer_size);
pca_initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
pca_initial_nn_params = [pca_initial_Theta1(:) ; pca_initial_Theta2(:)];


% initialize dct para
dct_initial_Theta1 = randInitializeWeights(dct_input_layer_size, hidden_layer_size);
dct_initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
dct_initial_nn_params = [dct_initial_Theta1(:) ; dct_initial_Theta2(:)];


% initialize dct and pca para
d_pca_initial_Theta1 = randInitializeWeights(d_pca_input_layer_size, hidden_layer_size);
d_pca_initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
d_pca_initial_nn_params = [d_pca_initial_Theta1(:) ; d_pca_initial_Theta2(:)];


% initialize lbp para
lbp_initial_Theta1 = randInitializeWeights(lbp_input_layer_size, hidden_layer_size);
lbp_initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
lbp_initial_nn_params = [lbp_initial_Theta1(:) ; lbp_initial_Theta2(:)];


% initialize lbp and dct para
l_dct_initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
l_dct_initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
l_dct_initial_nn_params = [l_dct_initial_Theta1(:) ; l_dct_initial_Theta2(:)];


%% =========================  Validation Curve  ===========================
%
% In the Validation Curve part, I will implement the valCurve function to
% test the performance of differnt parameter lambda, and dicide the value
% of the parameter we use in this neural network system.
%

% fprintf('\nTrain the neural network with different lambda...')
% %Initialize the parameter lambda
% 
% lambdaSet = [0.0005 0.001 0.05 0.07 0.1 0.3 0.5 1];   % find out the most suitablt lambda
% 
% % lambdaSet = [0.0001 0.0005 0.001 0.003 0.005];
% 
% % lambdaSet = 0.0001: 0.005:0.01;  % with lbp featrues
% 
% % e_train_lambda = zeros(size(lambdaSet));
% % e_val_lambda = zeros(size(lambdaSet));
% [e_train_lambda, e_val_lambda] = ...
%     valCurve(X_train, y_train,...
%     X_val, y_val, initial_nn_params,...
%     input_layer_size, hidden_layer_size, num_labels, lambdaSet);
% 
% 
% figure;
% plot(lambdaSet, e_train_lambda, lambdaSet, e_val_lambda);
% title('The learning Curve with different lambda')
% legend('Train', 'Cross Validation')
% xlabel('Lambda')
% ylabel('Cost')
% 
% 
% fprintf('# Lambda\tTrain Error\tCross Validation Error\n');
% for i = 1:length(lambdaSet)
%     fprintf('  \t%d\t\t%f\t%f\n', lambdaSet(i),...
%         e_train_lambda(i), e_val_lambda(i));
% end
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;


%% ==========================  Learning Curve  ============================
% 
% In the Learning Curve part, I will implement the LearningCurve function
% to test the performance of the model that I just created.
%

% lambda = 0.5; % Regarding to the result from the last part, set the
% % lambda to 0.5
% 
% [e_train, e_val] = ...
%     LearningCurve(X_train, y_train, X_val, y_val,...
%     initial_nn_params, input_layer_size, ...
%     hidden_layer_size, num_labels, lambda);
% 
% 
% e_train = nonzeros(e_train);
% e_val = nonzeros(e_val);
% figure;
% plot(size(X_train, 1) - length(e_train) + 1 : size(X_train, 1), ...
%     e_train, size(X_train, 1) - length(e_train) + 1 : size(X_train, 1),...
%     e_val);
% title('The learning Curve ')
% legend('Train', 'Cross Validation')
% xlabel('Number of training examples')
% ylabel('Cost')
% % axis([0 13 0 150])
% 
% 
% fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1 : length(e_train)
%     fprintf('  \t%d\t\t%f\t%f\n', size(X_train, 1) - length(e_train) + i,...
%         e_train(i), e_val(i));
% end
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% ============================= Training NN ==============================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

% % lambda = 0.5;
% % nn_params = trainNN(X_train(1:131,:), y_train(1:131,:),initial_nn_params, input_layer_size,...
% %     hidden_layer_size, num_labels, lambda);
% 
% %  After you have completed the assignment, change the MaxIter to a larger
% %  value to see how more training helps.
% options = optimset('MaxIter', 1500);
% 
% %  You should also try different values of lambda
% lambda = 0.001;
% 
% % Create "short hand" for the cost function to be minimized
% costFunction = @(p) nnCostFunction(p, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels,...
%                                    X_train(1:131,:), y_train(1:131,:),...
%                                    lambda);
% 
% % Now, costFunction is a function that takes in only one argument (the
% % neural network parameters)
% 
% tic;
% [nn_params, cost, exitflag] = fmincg(costFunction, initial_nn_params, options);
% toc;



% pca
fprintf('\nTraining Neural Network for pca... \n')
options = optimset('MaxIter', 1500);
lambda = 0.001;
pca_costFunction = @(p) nnCostFunction(p, ...
                                   pca_input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels,...
                                   pca_train(1:131,:), y_train(1:131,:),...
                                   lambda);
tic;
[pca_nn_params, pca_cost, pca_exitflag] = fmincg(pca_costFunction, pca_initial_nn_params, options);
toc;
pca_time = toc;

 % dct
fprintf('\nTraining Neural Network for dct... \n')
dct_costFunction = @(p) nnCostFunction(p, ...
                                   dct_input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels,...
                                   dct_train(1:131,:), y_train(1:131,:),...
                                   lambda);
tic;
[dct_nn_params, dct_cost, dct_exitflag] = fmincg(dct_costFunction, dct_initial_nn_params, options);
toc;
dct_time = toc;

% dct and pca
fprintf('\nTraining Neural Network for dct and pca... \n')
d_pca_costFunction = @(p) nnCostFunction(p, ...
                                   d_pca_input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels,...
                                   d_pca_train(1:131,:), y_train(1:131,:),...
                                   lambda);
tic;
[d_pca_nn_params, d_pca_cost, d_pca_exitflag] = fmincg(d_pca_costFunction, d_pca_initial_nn_params, options);
toc;
d_pca_time = toc;

% lbp
fprintf('\nTraining Neural Network for lbp... \n')
lbp_costFunction = @(p) nnCostFunction(p, ...
                                   lbp_input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels,...
                                   lbp_train(1:131,:), y_train(1:131,:),...
                                   lambda);
tic;
[lbp_nn_params, lbp_cost, lbp_exitflag] = fmincg(lbp_costFunction, lbp_initial_nn_params, options);
toc;
lbp_time = toc;

% lbp and dct
fprintf('\nTraining Neural Network for lbp and dct... \n')
l_dct_costFunction = @(p) nnCostFunction(p, ...
                                   l_dct_input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels,...
                                   l_dct_train(1:131,:), y_train(1:131,:),...
                                   lambda);
tic;
[l_dct_nn_params, l_dct_cost, l_dct_exitflag] = fmincg(l_dct_costFunction, l_dct_initial_nn_params, options);
toc;
l_dct_time = toc;


%% ============= Obtain Theta1 and Theta2 back from nn_params =============
% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                  hidden_layer_size, (input_layer_size + 1));
% 
% Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                  num_labels, (hidden_layer_size + 1));
             
% pca
pca_Theta1 = reshape(pca_nn_params(1:hidden_layer_size * (pca_input_layer_size + 1)), ...
                 hidden_layer_size, (pca_input_layer_size + 1));

pca_Theta2 = reshape(pca_nn_params((1 + (hidden_layer_size * (pca_input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% dct
dct_Theta1 = reshape(dct_nn_params(1:hidden_layer_size * (dct_input_layer_size + 1)), ...
                 hidden_layer_size, (dct_input_layer_size + 1));

dct_Theta2 = reshape(dct_nn_params((1 + (hidden_layer_size * (dct_input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% dct and pca
d_pca_Theta1 = reshape(d_pca_nn_params(1:hidden_layer_size * (d_pca_input_layer_size + 1)), ...
                 hidden_layer_size, (d_pca_input_layer_size + 1));

d_pca_Theta2 = reshape(d_pca_nn_params((1 + (hidden_layer_size * (d_pca_input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% lbp
lbp_Theta1 = reshape(lbp_nn_params(1:hidden_layer_size * (lbp_input_layer_size + 1)), ...
                 hidden_layer_size, (lbp_input_layer_size + 1));

lbp_Theta2 = reshape(lbp_nn_params((1 + (hidden_layer_size * (lbp_input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% lbp and dct
l_dct_Theta1 = reshape(l_dct_nn_params(1:hidden_layer_size * (l_dct_input_layer_size + 1)), ...
                 hidden_layer_size, (l_dct_input_layer_size + 1));

l_dct_Theta2 = reshape(l_dct_nn_params((1 + (hidden_layer_size * (l_dct_input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ========================= Implement Predict ============================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.
pred_Train = predict(Theta1, Theta2, X_train);
[a, y_Train] = max(y_train,[],2);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_Train == y_Train) * 100));


pred_Test = predict(Theta1, Theta2, X_test);
[b, y_Test] = max(y_test,[],2);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_Test == y_Test)) * 100);

pred_All = predict(Theta1, Theta2, X);
[c, y_All] = max(y, [], 2);
fprintf('\nAll Set Accuracy: %f\n', mean(double(pred_All == y_All)) * 100);


[a, y_Train] = max(y_train, [], 2);
[b, y_Test] = max(y_test,[],2);
[c, y_All] = max(y, [], 2);

pca_train_pred = predict(pca_Theta1, pca_Theta2, pca_train);
pca_test_pred = predict(pca_Theta1, pca_Theta2, pca_test);
fprintf('\npca Training Set Accuracy: %f\n', mean(double( pca_train_pred == y_Train)) * 100);
fprintf('\npca Test Set Accuracy: %f\n', mean(double( pca_test_pred == y_Test)) * 100);


dct_train_pred = predict(dct_Theta1, dct_Theta2, dct_train);
dct_test_pred = predict(dct_Thetea1, dct_Theta2, dct_test);
fprintf('\ndct Training Set Accuracy: %f\n', mean(double( dct_train_pred == y_Train)) * 100);
fprintf('\ndct Test Set Accuracy: %f\n', mean(double( dct_test_pred == y_Test)) * 100);



d_pca_train_pred = predict(d_pca_Theta1, d_pca_Theta2, d_pca_train);
d_pca_test_pred = predict(d_pca_Thetea1, d_pca_Theta2, d_pca_test);
fprintf('\npca and dct Training Set Accuracy: %f\n', mean(double( d_pca_train_pred == y_Train)) * 100);
fprintf('\npca and dct Test Set Accuracy: %f\n', mean(double( d_pca_test_pred == y_Test)) * 100);



lbp_train_pred = predict(lbp_Theta1, lbp_Theta2, lbp_train);
lbp_test_pred = predict(lbp_Thetea1, lbp_Theta2, lbp_test);
fprintf('\nlbp Training Set Accuracy: %f\n', mean(double( lbp_train_pred == y_Train)) * 100);
fprintf('\nlbp Test Set Accuracy: %f\n', mean(double( lbp_test_pred == y_Test)) * 100);



