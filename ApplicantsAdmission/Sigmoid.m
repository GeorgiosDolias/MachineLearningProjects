function g = Sigmoid(z)
% Computes the sigmoid of z.

g = zeros(size(z));


g = 1./(1+ exp(-z));
end
