function [J, grad] = LinearRegCostFunction(X, y, theta, lambda)
% Computes the cost of using theta as the parameter for linear regression  
% to fit the data points in X and y. Returns the cost and gradient 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


h = X*theta;


theta_reg = [0;theta(2:end, :);];
J = (1/(2*m)) * sum((h - y).^2) + (lambda/(2*m)) * (theta_reg' * theta_reg);
grad = (1/m) * X' * (h - y) + (lambda/m) * theta_reg;


grad = grad(:);

end
