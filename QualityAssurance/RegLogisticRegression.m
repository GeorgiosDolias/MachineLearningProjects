%% Regularised Logistic Regression with dataset from fabrication plant
%
%% Initialisation

clear ; close all; clc

%%  Loads Data
data = load('MicrochipData.txt');
% The results of the two microchips tests are saved in X matrix
% and the test scores (0 or 1) are saved in y vector   
X = data(:, [1, 2]); y = data(:, 3);

% Plot dataset
PlotData(X, y);

hold on;

% Labels and Legend
xlabel('Microchip Test 1','FontSize',18)
ylabel('Microchip Test 2','FontSize',18)

title("Quality assurance data from fabrication microchip plant",'FontSize',18)
% Specified in plot order
legend('y = 1', 'y = 0','FontSize',12,'TextColor','blue','LineWidth',1.0)
hold off;

%%  Adds polynomial features
% Adds polynomial features to our data matrix X
extentedX = MapFeature(X(:,1), X(:,2));

% Initializes fitting parameters
initial_theta = zeros(size(extentedX, 2), 1);

% Sets regularization hyperparameter lambda arbitrarily to 1;
lambda = 1;

% Computes initial cost and gradient for regularized logistic regression
[cost, grad] = CostFunctionReg(initial_theta, extentedX, y, lambda);

%%  Calculates optimal theta parameters
%
% Sets Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(CostFunctionReg(t, extentedX, y, lambda)), initial_theta, options);

% Plots Decision Boundary
PlotsDecisionBoundary(theta, extentedX, y)

hold on;
title(sprintf('Decision boundary with: lambda = %g', lambda),'FontSize',18)

% Labels and Legend
xlabel('Microchip Test 1','FontSize',18)
ylabel('Microchip Test 2','FontSize',18)

legend('y = 1', 'y = 0', 'Decision boundary','FontSize',12,'TextColor','blue','LineWidth',1.0)
hold off;

%%  Makes predictions
% Computes accuracy on our training set
p = Predict(theta, mapFeature(X(:,1), X(:,2)));

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
