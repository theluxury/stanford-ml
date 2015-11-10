function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

predicted_vector = X*theta;
difference_vector = y-predicted_vector;
difference_vector2 = difference_vector.^2;
J = sum(difference_vector2) / (2*m);


% =========================================================================

end
