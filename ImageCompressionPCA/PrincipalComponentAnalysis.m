%% Principal Component Analysis

%% Initialization
clear ; close all; clc

%% ================== Load Example Dataset  ===================

fprintf('Visualizing example dataset for PCA.\n\n');

load ('2Ddataset.mat');

%  Visualizes the example dataset
plot(X(:, 1), X(:, 2), 'bo');
xlabel('First Dimension','FontSize',18);
ylabel('Second Dimension','FontSize',18);
title('Example Dataset','FontSize',18);
axis([0.5 6.5 2 8]); axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Principal Component Analysis ===============

fprintf('\nRunning PCA on example dataset.\n\n');

%  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = FeatureNormalize(X);

%  Runs PCA
[U, S] = Pca(X_norm);

%  Computes mu, the mean of the each feature

%  Draws the eigenvectors centered at mean of data. These lines show the
%  directions of maximum variations in the dataset.
hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
legend('Data points','Eigenvectors','FontSize',12,'TextColor','black','LineWidth',1.0)
hold off;

fprintf('Top eigenvector: \n');
fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));
fprintf('\n(you should expect to see -0.707107 -0.707107)\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Dimension Reduction ===================
%  Maps the data onto the first k eigenvectors.

fprintf('\nDimension reduction on example dataset.\n\n');

%  Plots the normalized dataset (returned from pca)
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square
xlabel('First Dimension','FontSize',18);
ylabel('Second Dimension','FontSize',18);
title('Reduction of 2-dimensional dataset to 1 dimension','FontSize',18);

%  Projects the data onto K = 1 dimension
K = 1;
Z = ProjectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));
fprintf('\n(this value should be about 1.481274)\n\n');

X_rec  = RecoverData(Z, U, K);
fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));
fprintf('\n(this value should be about  -1.047419 -1.047419)\n\n');

%  Draws lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    DrawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
legend('Normalised dataset','Projected points','FontSize',12,'TextColor','black','LineWidth',1.0)
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Loading and Visualizing Face Data =============

fprintf('\nLoading face dataset.\n\n');

load ('facesImages.mat')

%  Displays the first 100 faces in the dataset
DisplayData(X(1:100, :));
title('Visualization of first 100 faces of dataset','FontSize',18)

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== PCA on Face Data: Eigenfaces  ===================

%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this might take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = FeatureNormalize(X);

%  Runs PCA
[U, S] = Pca(X_norm);

%  Visualizes the top 36 eigenvectors found
DisplayData(U(:, 1:36)');
title('Visualization of top 36 eigenvectors found','FontSize',18)
fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 

fprintf('\nDimension reduction for face dataset.\n\n');

K = 100;
Z = ProjectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% ==== Visualization of Faces after PCA Dimension Reduction ====
%  Projects images to the eigen space using the top K eigen vectors and 
%  visualizes only using those K dimensions
%  Compares to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec  = RecoverData(Z, U, K);

% Displays normalized data
subplot(1, 2, 1);
DisplayData(X_norm(1:100,:));
title('Original faces','FontSize',18);
axis square;

% Displays reconstructed data from only k eigenfaces
subplot(1, 2, 2);
DisplayData(X_rec(1:100,:));
title('Recovered faces','FontSize',18);
axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% === PCA for Visualization ===
%  One useful application of PCA is to use it to visualize high-dimensional
%  data.

close all; close all; clc


A = double(imread('bird_small.png'));


A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16; 
max_iters = 10;
initial_centroids = KMeansInitCentroids(X, K);
[centroids, idx] = RunkMeans(X, initial_centroids, max_iters);

%  Sample 1000 random indexes (since working with all the data is
%  too expensive. 
sel = floor(rand(1000, 1) * size(X, 1)) + 1;

%  Setups Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

%  Visualizes the data and centroid memberships in 3D
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
xlabel('First Dimension','FontSize',18);
ylabel('Second Dimension','FontSize',18);
title('Pixel dataset plotted in 3D. Color reflects clusters memberships','FontSize',18);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% === PCA for Visualization ===
% Uses PCA to project this cloud to 2D for visualization

% Subtracts the mean to use PCA
[X_norm, mu, sigma] = FeatureNormalize(X);

% PCA and project the data to 2D
[U, S] = Pca(X_norm);
Z = ProjectData(X_norm, U, 2);

% Plot in 2D
figure; axis square;
plotDataPoints(Z(sel, :), idx(sel), K);
xlabel('First Dimension','FontSize',18);
ylabel('Second Dimension','FontSize',18);

title('Pixel dataset plotted in 2D, using PCA.Color reflects clusters memberships','FontSize',18);
fprintf('Program paused. Press enter to continue.\n');
pause;
