%% Regularized Linear Regression and Bias-Variance

%% Initialization
clear ; close all; clc

%% =========== Loading and Visualizing Data =============

% Loads Training Data
fprintf('Loading and Visualizing Data ...\n')

load ('DamData.mat');

% m = Number of examples
m = size(X, 1);

% Plots training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)','FontSize',18);
ylabel('Water flowing out of the dam (y)','FontSize',18);
title('Historical records of water level changes and exported water','FontSize',18)
legend('points','FontSize',12,'TextColor','blue','LineWidth',1.0,'Location','southeast')
fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Regularized Linear Regression Cost =============
% Implements the cost function for regularized linear regression. 

theta = [1 ; 1];
lambda = 1;
[J, grad] = LinearRegCostFunction([ones(m, 1) X], y, theta, lambda);

fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Regularized Linear Regression Gradient =============

fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Train Linear Regression =============
%  TrainLinearReg function uses cost function to train 
%  regularized linear regression.


%  Train linear regression with lambda = 0
lambda = 0;
[theta] = TrainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)','FontSize',18);
ylabel('Water flowing out of the dam (y)','FontSize',18);
hold on;
plot(X, [ones(m, 1) X]*theta, 'g--', 'LineWidth', 2)
title('Visualisation of dataset and linear regression fit','FontSize',18)
legend('points','Fit','FontSize',12,'TextColor','blue','LineWidth',1.0,'Location','southeast')

hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Learning Curve for Linear Regression =============

%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" 
%

lambda = 0;
[error_train, error_val] = ...
    LearningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);
temp = 1:m;
p1 = plot(temp, error_train,'g','DisplayName','cos(2x)')
hold on
p2 = plot(temp, error_val,'b','DisplayName','cos(3x)')

title('Learning curve for linear regression','FontSize',18)

xlabel('Number of training examples','FontSize',18)
ylabel('Error','FontSize',18)
axis([0 13 0 150])

h = [p1; p2];
legend(h,'Train','Cross Validation','FontSize',18,'TextColor','black','LineWidth',1.0)
hold off




fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Feature Mapping for Polynomial Regression ========


p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = PolyFeatures(X, p);
[X_poly, mu, sigma] = FeatureNormalize(X_poly);  
X_poly = [ones(m, 1), X_poly];                   

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = PolyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];    

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = PolyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];       

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% =========== Learning Curve for Polynomial Regression ==========
%  Runs polynomial regression with lambda = 0. Try different values of
%  lambda

lambda = 1;
[theta] = TrainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
PlotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)','FontSize',18);
ylabel('Water flowing out of the dam (y)','FontSize',18);
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda),'FontSize',18);

figure(2);
[error_train, error_val] = ...
    LearningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve(lambda = %f)',lambda),'FontSize',18);
xlabel('Number of training examples','FontSize',18)
ylabel('Error','FontSize',18)
axis([0 13 0 100])
legend('Train', 'Cross Validation','FontSize',18,'TextColor','black','LineWidth',1.0)

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Validation for Selecting Lambda =============
%  Tests various values of lambda on a validation set. 
% Then, uses this to select the "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    ValidationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation','FontSize',18,'TextColor','black','LineWidth',1.0);
xlabel('lambda','FontSize',18);
ylabel('Error','FontSize',18);
title('Train and cross validation errors for different lambda values','FontSize',18)
fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
