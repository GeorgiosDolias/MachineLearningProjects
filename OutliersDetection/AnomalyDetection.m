%% Anomaly Detection

%% Initialization
clear ; close all; clc

%% ================== Load Example Dataset  ===================

%
%  Dataset consists of 2 network server statistics across
%  several machines: the latency and throughput of each machine.
%  It will help to find possibly faulty (or very fast) machines.
%

fprintf('Visualizing example dataset for outlier detection.\n\n');


load('ExampleDataset.mat');

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)','FontSize',18);
ylabel('Throughput (mb/s)','FontSize',18);
title('Visualisation of server statistics','FontSize',18);
legend('Dataset points','FontSize',18,'TextColor','black','LineWidth',1.0);
fprintf('Program paused. Press enter to continue.\n');
pause


%% ================== Estimate the dataset statistics ===================
%  For this exercise, we assume a Gaussian distribution for the dataset.
%
%  We first estimate the parameters of our assumed Gaussian distribution, 
%  then compute the probabilities for each of the points and then visualize 
%  both the overall distribution and where each of the points falls in 
%  terms of that distribution.
%
fprintf('Visualizing Gaussian fit.\n\n');

%  Estimates mu and sigma2
[mu sigma2] = EstimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) 
%  of X
p = MultivariateGaussian(X, mu, sigma2);

%  Visualize the fit
VisualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)','FontSize',18);
ylabel('Throughput (mb/s)','FontSize',18);
title('Gaussian fit for server statistics','FontSize',18);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Find Outliers ===================
%  Now you will find a good epsilon threshold using a cross-validation set
%  probabilities given the estimated Gaussian distribution
% 

pval = MultivariateGaussian(Xval, mu, sigma2);

[epsilon F1] = SelectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n');
fprintf('   (you should see a Best F1 value of  0.875000)\n\n');

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10,'DisplayName','Outliers');
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Multidimensional Outliers ===================


%  Loads the second dataset. 
load('MultiDDataset.mat');

%  Apply the same steps to the larger dataset
[mu sigma2] = EstimateGaussian(X);

%  Training set 
p = MultivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = MultivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon F1] = SelectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 1.38e-18)\n');
fprintf('   (you should see a Best F1 value of 0.615385)\n');
fprintf('# Outliers found: %d\n\n', sum(p < epsilon));
