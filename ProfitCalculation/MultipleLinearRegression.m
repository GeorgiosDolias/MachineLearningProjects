%% Linear regression with multiple variables
%

%% ================ Feature Normalization ================

% Clears and Closes Figures
clear ; close all; clc

fprintf('Loading data ...\n');

% Loads Data
data = load('HousingPrices.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Prints out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scales features and sets them to zero mean
fprintf('Normalizing Features ...\n');

[X, mu, sigma] = NormalizeFeatures(X);


% Adds intercept term to X
X = [ones(m, 1) X];


%% ================ Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Chooses some alpha value
alpha = 0.1;
num_iters = 1000;

% Inits Theta and Runs Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = MultiGradientDescent(X, y, theta, alpha, num_iters);

% Plots the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations','FontSize',18);
ylabel('Cost J','FontSize',18);
title('Computed cost over iterations','FontSize',18);

% Displays gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimates the price of a 1650 sq-ft, 3 br house

Pred_price =0;
secondprice = 1650*mu(1,1)/sigma(1,1);
thirdprice = 3*mu(1,2)/sigma(1,2);
Pred_price = [1 secondprice thirdprice] *theta;


fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], Pred_price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Normal Equations ================

fprintf('Solving with normal equations...\n');

% Loads Data
data = csvread('HousingPrices.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Adds intercept term to X
X = [ones(m, 1) X];

% Calculates the parameters from the normal equation
theta = NormalEquations(X, y);

% Displays normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimates the price of a 1650 sq-ft, 3 br house

Pred_price_n = 0; 

Pred_price_n = [1 1650 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], Pred_price_n);

