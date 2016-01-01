function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X]; % add bias unit
hidden_layer = X*Theta1.'; % mx401 * 401x25 = mx25
hidden_layer_sigmoid = sigmoid(hidden_layer); % take signmoid duh. 
hidden_layer_sigmoid = [ones(m,1) hidden_layer_sigmoid]; % add one for bias unit
output_layer = hidden_layer_sigmoid*Theta2.'; %mx26  * 26x10 = mx10
output_layer_sigmoid = sigmoid(output_layer);

% probably easier just with for loop
for i = 1:m
    yi = zeros(1, num_labels);
    yi(y(i)) = 1;
    J = J + (1/m) * sum(-yi.*log(output_layer_sigmoid(i,:)) - (1-yi).*log(1-output_layer_sigmoid(i,:)));
end

% then regularize it...
theta1_copy = Theta1;
theta2_copy = Theta2;
theta1_copy(:,1)= [];
theta2_copy(:,1) = [];
[s1,~] = sumsqr(theta1_copy);
[s2,~] = sumsqr(theta2_copy);
regularized_term = lambda * (s1 + s2) / (2 * m);
J = J + regularized_term;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for i = 1:m
    yi = zeros(1, num_labels);
    yi(y(i)) = 1;
    sigma3i = (output_layer_sigmoid(i,:) - yi).'; % make it into column
    placeholder = Theta2.'*sigma3i; % to keep it consistent with the instructions.
    z_2 = hidden_layer(i,:).';
    z_2 = [1 ; z_2]; % add a bias unit
    placeholder2 = sigmoidGradient(z_2);
    sigma2i = placeholder.*placeholder2;
    sigma2i = sigma2i(2:end); % remove the first column
    a_1 = X(i,:).'; % already has bias unit. Also, make into column for consistentcy with instruction.
    Theta1_grad = Theta1_grad + sigma2i*(a_1.');
    a_2 = hidden_layer_sigmoid(i,:).'; % first row, then, uh, make it a column. 
    Theta2_grad = Theta2_grad + sigma3i*(a_2.');
end

Theta1_grad = Theta1_grad./m;
Theta2_grad = Theta2_grad./m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1 = Theta1.*(lambda / m);
Theta2 = Theta2.*(lambda / m);
% delete first column, then add 0 column
Theta1(:,1) = [];
Theta1 = [zeros(size(Theta1,1),1) Theta1];
Theta2(:,1) = [];
Theta2 = [zeros(size(Theta2,1),1) Theta2];

% add it
Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
