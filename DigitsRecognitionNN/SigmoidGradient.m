function g = SigmoidGradient(z)
% Computes the gradient of the sigmoid function
% evaluated at z. This should work regardless if z is a matrix or a
% vector. 

g = zeros(size(z));

g = Sigmoid(z).*(1-Sigmoid(z));
end
