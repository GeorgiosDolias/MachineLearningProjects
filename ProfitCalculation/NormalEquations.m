function [theta] = NormalEquations(X, y)
% Computes the closed-form solution to linear 
% regression using the normal equations.

theta = zeros(size(X, 2), 1);


theta = (X'*X)\(X'*y);

end
