function J = CostFun(X, y, theta)
% Computes the cost of using theta as the
% parameter for linear regression to fit the data points in X and y

% Initializes some useful values
m = length(y); % number of training examples

J = 0;
sum = 0;

h=X*theta;

for i=1:m
    sum= sum + (h(i)-y(i))^2;
end
    
J = (1/(2*m))*sum;

end
