function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
sum =0;


for j=2:length(theta);
    sum = sum + theta(j)^2;
end

%for i=1,m;
%    sum = sum + (-y(i)*log(h(i))-(1-y(i))*log(1-h(i)));
%end

%J = (1/m)* sum;

% Vectorized implementation

J = (1/m)*(-y'*log(h)-(1-y)'*log(1-h))+ (lambda/(2*m))*sum;


% Gradient

for j=1:size(theta,1);
    sum2=0;
    %j
    for i=1:m;
        sum2 = sum2 + (h(i)-y(i))*X(i,j);
    end
    if(j==1);
        grad(j) = (1/m)*sum2;
    else
        grad(j) = (1/m)*sum2 + lambda*theta(j)/m;
    end
end


% =============================================================

end
