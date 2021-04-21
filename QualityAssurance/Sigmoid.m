function g = Sigmoid(z)

% Initialises values of Sigmoid function 
g = zeros(size(z));

g = 1./(1+ exp(-z));

end
