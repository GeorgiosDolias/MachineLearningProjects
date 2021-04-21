%% Logistic Regression
%

%% Initialization
clear ; close all; clc

%% Loads Data

data = load('ExamsScores.txt');

%  The first two columns contain the exam scores and the third column
%  contains the label.
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Plotting ====================


fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

PlotData(X, y);

% Puts some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score','FontSize',18)
ylabel('Exam 2 score','FontSize',18)

title('Exams Scores','FontSize',18)
% Specified in plot order
legend('Admitted', 'Not admitted','FontSize',18,'TextColor','blue','LineWidth',1.0)
legend('boxon')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============ Compute Cost and Gradient ============

%  Setups the data matrix appropriately, and adds ones for the intercept term
[m, n] = size(X);

% Adds intercept term to x and X_test
X = [ones(m, 1) X];

% Initializes fitting parameters
initial_theta = zeros(n + 1, 1);

% Computes and displays initial cost and gradient
[cost, grad] = CostFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);


% Computes and displays cost and gradient with non-zero theta
test_theta = [-24; 0.2; 0.2];
[cost, grad] = CostFunction(test_theta, X, y);

fprintf('\nCost at test theta: %f\n', cost);

fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);


fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============= Optimizing using fminunc  =============
%  A built-in function (fminunc) is used to find the
%  optimal parameters theta.

%  Sets options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Runs fminunc to obtain the optimal theta

[theta, cost] = ...
	fminunc(@(t)(CostFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);

fprintf('theta: \n');
fprintf(' %f \n', theta);


% Plots Boundary
PlotDecisionBoundary(theta, X, y);

% Puts some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score','FontSize',18)
ylabel('Exam 2 score','FontSize',18)
title('Decision boundary based on optimal parameters','FontSize',18)
% Specified in plot order
legend('Admitted', 'Not admitted','FontSize',18,'TextColor','blue','LineWidth',1.0)
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Predict and Accuracies ==============

%  Predicts probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

prob = Sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob);

% Computes accuracy on our training set
p = Predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
