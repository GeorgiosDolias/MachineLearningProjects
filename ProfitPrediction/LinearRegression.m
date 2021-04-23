%% Linear Regression with one variable

%% Initialization
clear ; close all; clc

%% ======================= Plotting =======================
fprintf('Plotting Data ...\n')
data = load('RestaurantData.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

% Plots Data
CreatePlot(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Cost and Gradient descent ===================

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01; %     Learning rate

fprintf('\nTesting the cost function ...\n')
% Computes and display initial cost
J = CostFun(X, y, theta);
fprintf('Cost computed = %f\n', J);


fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')
% runs gradient descent
theta = GradDescent(X, y, theta, alpha, iterations);

% prints theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

% Plots the linear fit
hold on; % keeps previous plot visible
plot(X(:,2), X*theta, '-');
legend('Training data', 'Linear regression','Location','southeast')
%title('Input data and linear regression fit')

hold off % don't overlay any more plots on this figure

% Predicts values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, predicted profit is %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, predicted profit is %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initializes J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fills out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = CostFun(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0','FontSize',18); ylabel('\theta_1','FontSize',18);
title('Cost J visualisation','FontSize',18);
% Contour plot
figure;
% Plots J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0','FontSize',18); ylabel('\theta_1','FontSize',18);
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
title('Contour Plot of Cost function','FontSize',18);
