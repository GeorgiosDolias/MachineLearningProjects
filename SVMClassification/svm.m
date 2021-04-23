%% Support Vector Machines
%

%% Initialization
clear ; close all; clc

%% =============== Loading and Visualizing Data ================

fprintf('Loading and Visualizing Data ...\n')

% Loads from ex6data1: 
% You will have X, y in your environment
load('Dataset1.mat');

% Plot training data
PlotsData(X, y,'dataset 1');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Training Linear SVM ====================
% Trains a linear SVM on the dataset and plots the
% decision boundary learned.
%

% Load from ex6data1: 
% You will have X, y in your environment
load('Dataset1.mat');

fprintf('\nTraining Linear SVM ...\n')

% Change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 1;
model = SvmTrain(X, y, C, @linearKernel, 1e-3, 20);
VisualizeBoundaryLinear(X, y, model,'dataset1');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Implementing Gaussian Kernel ===============
% Implements the Gaussian kernel to use
% with the SVM.
%
fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = GaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Visualizing Dataset 2 ================


fprintf('Loading and Visualizing Dataset 2 ...\n')

% Load from ex6data2: 

load('Dataset2.mat');

% Plot training data
PlotsData(X, y,'dataset 2');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Training SVM with RBF Kernel (Dataset 2) ==========
% After implemented the kernel, it is used to train the 
% SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');


load('Dataset2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% The tolerance and max_passes lower are set here so that the code will run
% faster. 
model= SvmTrain(X, y, C, @(x1, x2) GaussianKernel(x1, x2, sigma)); 
VisualizeBoundary(X, y, model,'dataset 2');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Visualizing Dataset 3 ================


fprintf('Loading and Visualizing Dataset 3 ...\n')

load('Dataset3.mat');

% Plot training data
PlotsData(X, y,'dataset 3');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Training SVM with RBF Kernel (Dataset 3) ==========

load('Dataset3.mat');

% Try different SVM Parameters here
[C, sigma] = Dataset3Params(X, y, Xval, yval);

% Train the SVM
model= SvmTrain(X, y, C, @(x1, x2) GaussianKernel(x1, x2, sigma));
VisualizeBoundary(X, y, model,'dataset 3');

fprintf('Program paused. Press enter to continue.\n');
pause;

