function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
predicted_vector = X*theta;
difference_vector = predicted_vector - y;
difference_vector2 = difference_vector.^2;
J = sum(difference_vector2) / (2*m);
% kind of a hacky way to ignore first term, but sure.
J = J + (lambda / (2*m)) * (sumsqr(theta) - theta(1)^2);  

grad(1) = (1/m) * sum(difference_vector.* X(:, 1)); % first one no reg term
for i = 2:size(theta)
    grad(i) = (1/m) * sum(difference_vector.* X(:, i)) + (lambda / m) * theta(i);
end

% =========================================================================

grad = grad(:);

end
