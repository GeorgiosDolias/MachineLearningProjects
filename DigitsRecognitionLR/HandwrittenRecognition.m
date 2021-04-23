%% Machine Learning One-vs-all

%% Initialization
clear ; close all; clc

%% Setups the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that "0" is mapped  to label 10)

%% =========== Loading and Visualizing Data =============
%   Dataset contains handwritten digits.

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('HandwrittenDigits.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

DisplayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Vectorize Logistic Regression ============
%  Regularized logistic regression implementation is vectorized. 


% Test case for lrCostFunction
fprintf('\nTesting LRCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = LRCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Gradients:\n');
fprintf(' %f \n', grad);

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============ One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = OneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Predict for One-Vs-All ================

pred = PredictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

