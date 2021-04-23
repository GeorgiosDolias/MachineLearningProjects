function [lambda_vec, error_train, error_val] = ...
    ValidationCurve(X, y, Xval, yval)
% Returns the train and validation errors (in error_train, error_val)
% for different values of lambda. 

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';


error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);


for i = 1:length(lambda_vec)
   lambda = lambda_vec(i);
   theta = TrainLinearReg(X,y,lambda);
   
   error_train(i) = ComputeCost(X, y, theta);
   error_val(i) = ComputeCost(Xval, yval, theta);
end


end
