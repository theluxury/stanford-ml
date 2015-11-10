function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features
J_history = zeros(num_iters, 1);
theta_temp = zeros(n, 1);

for iter = 1:num_iters
    for feature = 1:n
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    theta_temp(feature) = theta(feature) - alpha * (1/m) * sum((X * theta - y).* X(:, feature));
    end
    theta = theta_temp;
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
