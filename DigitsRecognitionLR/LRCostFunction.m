function [J, grad] = LRCostFunction(theta, X, y, lambda)
% Computes the cost of using theta as the parameter for regularized 
% logistic regression and the gradient of the cost w.r.t. to the parameters. 

% Initializes some useful values
m = length(y); % number of training examples

 
J = 0;
grad = zeros(size(theta));


sum2=0;

h = sigmoid(X*theta);

sum2 =sum(theta(2:length(theta)).^2);


% Vectorized implementation

J = (1/m)*(-y'*log(h)-(1-y)'*log(1-h))+ (lambda/(2*m))*sum2;


% Vectorized Gradient

grad =(1/m)* X'*(h-y);

temp= theta;
temp(1) = 0;

grad = grad + (lambda/m)*temp;  

grad = grad(:);

end
