function [error_train, error_val] = ...
    valCurve(X, y, Xval, yval,initial_nn_params,...
    input_layer_size, hidden_layer_size, num_labels, lambda)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%       
%   
%   Note: The input parameter lambda is a vector with each element to test.
%


% You need to return these variables correctly.
error_train = zeros(size(lambda));
error_val = zeros(size(lambda));

for i = 1:length(lambda)
    lambdaTest = lambda(i);
    % 
    % Compute train / val errors when training the neural network
    % with regularization parameter lambda
    % Store the result in error_train(i)
    % and error_val(i)
    %
    
    % compute the parameters of the neural network theta
    nn_params = trainNN(X, y, initial_nn_params,...
    input_layer_size, hidden_layer_size, num_labels,lambdaTest); 

    % compute the error corresponds to certain lambda
    error_train(i) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels,X, y, lambdaTest);
    
    error_val(i) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels,Xval, yval,lambdaTest);
end

% =========================================================================

end
