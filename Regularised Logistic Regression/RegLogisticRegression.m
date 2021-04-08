%% Regularised Logistic Regression with dataset from fabrication plant
%
%% Initialisation

clear ; close all; clc

%%  Loads Data
data = load('ex2data2.txt');
%   The results of the two microchips tests are saved in X matrix
%   and the test scores (0 or 1) are saved in y vector   
X = data(:, [1, 2]); y = data(:, 3);

%   Function PlotData is called
PlotData(X, y);

%   Plot Details
hold on;

%   Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

title("Quality assurance data from fabrication microchip plant")
%   Specified in plot order
legend('y = 1', 'y = 0')
hold off;

%%  Adds polynomial features
%   Adds polynomial features to our data matrix X
extentedX = mapFeature(X(:,1), X(:,2));

%   Initialize fitting parameters
initial_theta = zeros(size(extentedX, 2), 1);

%   Set regularization hyperparameter lambda arbitrarily to 1;
lambda = 1;

%   Compute initial cost and gradient for regularized logistic regression
[cost, grad] = CostFunctionReg(initial_theta, extentedX, y, lambda);

%%  Calculates optimal theta parameters
%
%   Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

%   Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(CostFunctionReg(t, extentedX, y, lambda)), initial_theta, options);

%   Plot Decision Boundary
PlotDecisionBoundary(theta, extentedX, y)

hold on;
title(sprintf('Decision boundary with: lambda = %g', lambda))

%   Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

%%  Makes predictions
%   Computes accuracy on our training set
p = Predict(theta, mapFeature(X(:,1), X(:,2)));

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
