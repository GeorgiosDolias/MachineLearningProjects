function PlotDataPoints(X, idx, K)
% Plots data points in X, coloring them so that those 
% with the same index assignments in idx have the same color

% Create palette
palette = hsv(K + 1);
colors = palette(idx, :);

% Plot the data
scatter(X(:,1), X(:,2), 15, colors);

end