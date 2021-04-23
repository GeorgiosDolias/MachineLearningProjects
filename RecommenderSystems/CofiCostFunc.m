function [J, grad] = CofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
% Returns the cost and gradient for the
% collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%


size(X);
size(Theta');

%   Cost function calculation 

sum2 = 0;
h=X*Theta';

size(h);
size(Y);


cost= (h-Y).^2;

sum2 = sum(cost(R == 1));


J = (1/2)*sum2;

% Reguarized cost

J = J + lambda* sum(sum(Theta.^2))/2 + lambda * sum(sum(X.^2))/2;


% Gradients calculation


X_grad = ((X*Theta'-Y).*R)*Theta;
Theta_grad = ((X*Theta'-Y).*R)'*X;


% Regularized Gradient

X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda *Theta;

grad = [X_grad(:); Theta_grad(:)];

end
