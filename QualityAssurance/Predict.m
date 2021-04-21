function p = Predict(theta, X)
%   Computes the predictions for X using a 
%   threshold at 0.5.

m = size(X, 1); % Number of training examples

% Initialises predictions vector
p = zeros(m, 1);

threshold = 0.5;


h = Sigmoid(X*theta);

for j=1:m
    if h(j)< threshold
        p(j) =0;
    else
        p(j) = 1;
    end    
end

end
