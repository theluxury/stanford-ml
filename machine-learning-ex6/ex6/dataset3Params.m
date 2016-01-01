function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cVector = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaVector = [0.01 0.03 0.1 0.3 1 3 10 30];
errors = zeros(length(cVector) * length(sigmaVector), 1);
for i = 1:length(cVector)
    for j = 1:length(sigmaVector)
        model= svmTrain(X, y, cVector(i), @(x1, x2) gaussianKernel(x1, x2, sigmaVector(j)));
        predictions = svmPredict(model, Xval);
        errors((i - 1)*length(cVector) + j) = mean(double(predictions ~= yval));
    end
end

[~,minIndex] = min(errors);
C = cVector(floor((minIndex - 1)/length(cVector)) +1);
sigma = sigmaVector(mod(minIndex, length(sigmaVector)));

% =========================================================================

end
