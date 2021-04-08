function [J, grad] = CostFunctionReg(theta, X, y, lambda)
%   Computes cost and gradient for logistic regression with regularization
%   using theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


h = Sigmoid(X*theta);
sum =0;


for j=2:length(theta);
    sum = sum + theta(j)^2;
end


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

end
