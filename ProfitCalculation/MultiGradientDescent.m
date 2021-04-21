function [theta, J_history] = MultiGradientDescent(X, y, theta, alpha, num_iters)
% Updates theta by
% taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
      
    h=X*theta;
    
    for j=1:size(X,2)     % Features = Columns        
        
        theta(j) = theta(j) -alpha*(1/m)*sum((h-y) .* X(:,j));     
         
    end
    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
