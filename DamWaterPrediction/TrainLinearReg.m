function [theta] = TrainLinearReg(X, y, lambda)
% Trains linear regression using
% the dataset (X, y) and regularization parameter lambda. Returns the
% trained parameters theta.

% Initializes Theta
initial_theta = zeros(size(X, 2), 1); 

% Creates "short hand" for the cost function to be minimized
costFunction = @(t) LinearRegCostFunction(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');

% Minimizes using fmincg
theta = Fmincg(costFunction, initial_theta, options);

end
