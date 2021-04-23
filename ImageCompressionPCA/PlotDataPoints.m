function PlotDataPoints(X, idx, K)
% Plots data points in X, coloring them so that those 
% with the same index assignments in idx have the same color

% Creates palette
palette = hsv(K + 1);
colors = palette(idx, :);

% Plots the data
scatter(X(:,1), X(:,2), 15, colors);

end
