function PlotProgresskMeans(X, centroids, previous, idx, K, i)
% Plots the data points with colors assigned to each centroid. 
% With the previous centroids, it also plots a line between 
% the previous locations and current locations of the centroids.


% Plot the examples
PlotDataPoints(X, idx, K);

% Plot the centroids as black x's
plot(centroids(:,1), centroids(:,2), 'x', ...
     'MarkerEdgeColor','k', ...
     'MarkerSize', 10, 'LineWidth', 3);

% Plot the history of the centroids with lines
for j=1:size(centroids,1)
    drawLine(centroids(j, :), previous(j, :));
end

% Title
title(sprintf('Creating clusters in iteration number %d', i),'FontSize',18)
xlabel('First Dimension','FontSize',18)
ylabel('Second Dimension','FontSize',18)
legend('Dataset points','Centroids','FontSize',18,'TextColor','black','LineWidth',1.0)

end

