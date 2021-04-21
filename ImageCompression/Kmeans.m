%% K-Means Clustering

%% Initialization
clear ; close all; clc

%% ================= Find Closest Centroids ====================

fprintf('Finding closest centroids.\n\n');

% Loads an example dataset 
load('Dataset.mat');

% Selects an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Finds the closest centroids for the examples using the
% initial_centroids
idx = FindClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ===================== Compute Means =========================

fprintf('\nComputing centroids means.\n\n');

%  Computes means based on the closest centroids found in the previous part.
centroids = ComputeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== K-Means Clustering ======================

fprintf('\nRunning K-Means clustering on example dataset.\n\n');

% Loads example dataset
load('Dataset.mat');

% Settings for running K-Means
K = 3;
max_iters = 10;

% For consistency, here we set centroids to specific values
% but in practice you want to generate them automatically, such as by
% settings them to be random examples (as can be seen in
% kMeansInitCentroids).
initial_centroids = [3 3; 6 2; 8 5];

% Runs K-Means algorithm. The 'true' at the end tells our function to plot
% the progress of K-Means
[centroids, idx] = RunkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= K-Means Clustering on Pixels ===============

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Loads an image of a bird
A = double(imread('bird_small.png'));


A = A / 255; % Divides by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

% Reshapes the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
% This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% Runs K-Means algorithm on this data
% Try different values of K and max_iters here
K = 16; 
max_iters = 10;

% When using K-Means, it is important the initialize the centroids
% randomly. 

initial_centroids = KMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = RunkMeans(X, initial_centroids, max_iters);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Image Compression ======================

fprintf('\nApplying K-Means to compress an image.\n\n');

% Finds closest cluster members
idx = FindClosestCentroids(X, centroids);


% Recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
X_recovered = centroids(idx,:);

% Reshapes the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Displays the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original','FontSize',12);

% Displays compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K),'FontSize',12);


fprintf('Program paused. Press enter to continue.\n');
pause;