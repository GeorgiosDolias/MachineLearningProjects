function [J, grad] = CostFunction(theta, X, y)
% Computes the cost of using theta as the
% parameter for logistic regression and the gradient of the cost
% w.r.t. to the parameters.

% Initializes some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));


h = sigmoid(X*theta);
sum =0;

% Vectorized implementation

J = (1/m)*(-y'*log(h)-(1-y)'*log(1-h));


for j=1:size(theta,1);
    sum2=0;
    %j
    for i=1:m;
        sum2 = sum2 + (h(i)-y(i))*X(i,j);
    end
    grad(j) = (1/m)*sum2;
end

end
