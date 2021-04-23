function [theta, J_history] = GradDescent(X, y, theta, alpha, num_iters)
% Updates theta by taking num_iters gradient steps with learning rate alpha


m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    sum1 = 0;
    sum2 = 0;
    
    h=X*theta;
    
    for i=1:m
        sum1 = sum1 + (h(i)-y(i))*X(i,1);
        sum2 = sum2 + (h(i)-y(i))*X(i,2);
    end
    
    temp1 =  theta(1) - alpha*(1/m)*sum1;
    temp2 = theta(2) - alpha*(1/m)*sum2;
    theta(1) =temp1;
    theta(2) =temp2;

    % Saves the cost J in every iteration    
    J_history(iter) = CostFun(X, y, theta);
    if iter==150
        fprintf('%f\n', J_history(iter));
    end

end

end
