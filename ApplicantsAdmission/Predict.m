function p = Predict(theta, X)
% Computes the predictions for X using a 
% threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples


p = zeros(m, 1);



h = sigmoid(X*theta);

for j=1:m;
    if h(j)< 0.5;
        p(j) =0;
    else
        p(j) = 1;
    end    
end


end
