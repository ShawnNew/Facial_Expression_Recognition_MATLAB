function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, input_layer_size,
%   hidden_layer_size, num_labels, X, y, lambda) computes the cost and  the
%   gradient of the neural network.
%   
%   The parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
%   Note: The input parameter is a matrix with each column a num_labels'
%   vector, and the correspondant position of the vector is 1 only if that
%   example is defined as such label. Naturally, the other position of the
%   vector is set to 0.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...%The length of the first parameters
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%% *********************Part1 Feedforward the Neural Network***************
%**************************Feedforward the neural network******************
X = [ones(size(X,1),1), X];   % add the extra bias units.
z2 = Theta1 * X';
a2 = sigmoid(z2);    % the output of the hidden layer.
a2 = [ones(1, size(a2,2)); a2];% add the extra bias units.
z3 = Theta2 * a2;
a3 = sigmoid(z3);     % compute the prediction.
h_Theta = a3;
% %****************Convert the ys into matrix********************************
% y_m = zeros(m, num_labels);
% for k = 1:m
%    y_m(k,:) = zeros(1, max(y));
%    y_m(k, y(k)) = 1;
% end
%****************Compute the cost function*********************************
for i = 1:m
   J = J + (-1/m) * sum((y(i,:).* log(h_Theta(:,i))' ...
       +(1-y(i,:)).* log(1-h_Theta(:,i))')); 
end

Theta1_to_sum = reshape(Theta1(:, 2:end), size(Theta1, 1),...
    size(Theta1,2)-1);
Theta2_to_sum = reshape(Theta2(:, 2:end), size(Theta2,1),...
    size(Theta2,2)-1);


%% ********************Part2 Implement backprop algorithm******************
grad_temp1 = zeros(size(Theta1_grad));
grad_temp2 = zeros(size(Theta2_grad));
for t = 1:m
   delta3 = a3(:,t) - y(t,:)';
   delta2 = (Theta2(:,2:end)' * delta3).* (sigmoidGradient(z2(:,t)));
   grad_temp1 = grad_temp1 + delta2 * X(t,:);
   grad_temp2 = grad_temp2 + delta3 * a2(:,t)';
end
Theta1_grad = (1/m) * grad_temp1;
Theta2_grad = (1/m) * grad_temp2;

%% ******************************Regularization****************************
% Cost Function with regularized parameters
J = J + (lambda/(2 * m)) * (sum(sum((Theta1_to_sum.^2)))...
    + sum(sum((Theta2_to_sum.^2)))); 
% Regularize the gradient
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ...
    (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ...
    (lambda/m) * Theta2(:, 2:end);
 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
