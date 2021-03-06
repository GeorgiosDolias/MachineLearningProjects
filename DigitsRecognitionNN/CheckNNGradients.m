function CheckNNGradients(lambda)
% Creates a small neural network to check the
% backpropagation gradients, it will output the analytical gradients
% produced by your backprop code and the numerical gradients (computed
% using computeNumericalGradient). These two gradient computations should
% result in very similar values.


if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% 'random' test data
Theta1 = DebugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = DebugInitializeWeights(num_labels, hidden_layer_size);
% Reusing DebugInitializeWeights to generate X
X  = DebugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';

% Unrolls parameters
nn_params = [Theta1(:) ; Theta2(:)];

% Short hand for cost function
costFunc = @(p) NNCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);

[cost, grad] = costFunc(nn_params);
numgrad = ComputeNumericalGradient(costFunc, nn_params);

% Visually examines the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['The above two columns you get should be very similar.\n' ...
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

% Evaluates the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end
