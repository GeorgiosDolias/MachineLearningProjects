%% Neural Network Learning

%% Initialization
clear ; close all; clc

%% Setups the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that "0" has been mapped to label 10)

%% =========== Loading and Visualizing Data =============

% Loads Training Data
fprintf('Loading and Visualizing Data ...\n')

load('HandwrittenDigits.mat');
m = size(X, 1);

% Randomly selects 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

DisplayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Loading Parameters ================
% Loads some pre-initialized neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Loads the weights into variables Theta1 and Theta2
load('InitialWeights.mat');

% Unrolls parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Compute Cost (Feedforward) ================
%  First, feedforward part of the neural network is implemented
%  that returns the cost only. 
%
fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter
lambda = 0;

J = NNCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =============== Implement Regularization ===============


fprintf('\nChecking Cost Function (with Regularization) ... \n')

% Weight regularization parameter set to 1 here).
lambda = 1;

J = NNCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Sigmoid Gradient  ================


fprintf('\nEvaluating sigmoid gradient...\n')

g = SigmoidGradient([-1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Initializing Pameters ================
%  Initializes the weights of the neural network


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = RandInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = RandInitializeWeights(hidden_layer_size, num_labels);

% Unrolls parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Implement Backpropagation ===============
%  Implements the backpropagation algorithm for the neural network.

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
CheckNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Implement Regularization ===============

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Checks gradients by running checkNNGradients
lambda = 3;
CheckNNGradients(lambda);


debug_J  = NNCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Training NN ===================
%  To train the neural network, "fmincg" is now used, which
%  is a function which works similarly to "fminunc". These
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  Changes the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  Tries different values of lambda
lambda = 1;

% Creates "short hand" for the cost function to be minimized
costFunction = @(p) NNCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = Fmincg(costFunction, initial_nn_params, options);

% Obtains Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Visualize Weights =================
%  Displays the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

DisplayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Implement Predict =================


pred = PredictLabel(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


