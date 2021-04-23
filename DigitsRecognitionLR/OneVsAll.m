function [all_theta] = OneVsAll(X, y, num_labels, lambda)
% Trains multiple logistic regression classifiers and returns all
% the classifiers in a matrix all_theta, where the i-th row of all_theta 
% corresponds to the classifier for label i


% Some useful variables
m = size(X, 1);
n = size(X, 2);

% initialisation
all_theta = zeros(num_labels, n + 1);

% Adds ones to the X data matrix
X = [ones(m, 1) X];


for c=1:num_labels
    
    % Initialize fitting parameters
    initial_theta = zeros(n + 1, 1);
    
    %  Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50);

    %  Run fminunc to obtain the optimal theta
    [theta] = ...
          fmincg (@(t)(LRCostFunction(t, X, (y == c), lambda)), ...
                 initial_theta, options);
    theta;         
    all_theta(c,:) = theta; 
end

end
