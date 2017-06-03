function [nn_params] = trainNN(X, y, initial_nn_params,...
    input_layer_size, hidden_layer_size, num_labels, lambda)
%TRAINNN Trains neural network given a dataset (X, y) and a
%regularization parameter lambda
%   [theta] = TRAINNN (X, y, lambda) trains neural network using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.
%

% Initialize the parameters


% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');
% Minimize using fmincg
tic;
nn_params = fmincg(costFunction, initial_nn_params, options);
toc;


end